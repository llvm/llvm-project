//===-- FIRBuilder.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Allocatable.h"
#include "flang/Optimizer/Builder/Runtime/Assign.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/Utils.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MD5.h"
#include <optional>

static llvm::cl::opt<std::size_t>
    nameLengthHashSize("length-to-hash-string-literal",
                       llvm::cl::desc("string literals that exceed this length"
                                      " will use a hash value as their symbol "
                                      "name"),
                       llvm::cl::init(32));

aiir::func::FuncOp
fir::FirOpBuilder::createFunction(aiir::Location loc, aiir::ModuleOp module,
                                  llvm::StringRef name, aiir::FunctionType ty,
                                  aiir::SymbolTable *symbolTable) {
  return fir::createFuncOp(loc, module, name, ty, /*attrs*/ {}, symbolTable);
}

aiir::func::FuncOp
fir::FirOpBuilder::createRuntimeFunction(aiir::Location loc,
                                         llvm::StringRef name,
                                         aiir::FunctionType ty, bool isIO) {
  aiir::func::FuncOp func = createFunction(loc, name, ty);
  func->setAttr(fir::FIROpsDialect::getFirRuntimeAttrName(), getUnitAttr());
  if (isIO)
    func->setAttr("fir.io", getUnitAttr());
  return func;
}

aiir::func::FuncOp
fir::FirOpBuilder::getNamedFunction(aiir::ModuleOp modOp,
                                    const aiir::SymbolTable *symbolTable,
                                    llvm::StringRef name) {
  if (symbolTable)
    if (auto func = symbolTable->lookup<aiir::func::FuncOp>(name)) {
#ifdef EXPENSIVE_CHECKS
      assert(func == modOp.lookupSymbol<aiir::func::FuncOp>(name) &&
             "symbolTable and module out of sync");
#endif
      return func;
    }
  return modOp.lookupSymbol<aiir::func::FuncOp>(name);
}

aiir::func::FuncOp
fir::FirOpBuilder::getNamedFunction(aiir::ModuleOp modOp,
                                    const aiir::SymbolTable *symbolTable,
                                    aiir::SymbolRefAttr symbol) {
  if (symbolTable)
    if (auto func = symbolTable->lookup<aiir::func::FuncOp>(
            symbol.getLeafReference())) {
#ifdef EXPENSIVE_CHECKS
      assert(func == modOp.lookupSymbol<aiir::func::FuncOp>(symbol) &&
             "symbolTable and module out of sync");
#endif
      return func;
    }
  return modOp.lookupSymbol<aiir::func::FuncOp>(symbol);
}

fir::GlobalOp
fir::FirOpBuilder::getNamedGlobal(aiir::ModuleOp modOp,
                                  const aiir::SymbolTable *symbolTable,
                                  llvm::StringRef name) {
  if (symbolTable)
    if (auto global = symbolTable->lookup<fir::GlobalOp>(name)) {
#ifdef EXPENSIVE_CHECKS
      assert(global == modOp.lookupSymbol<fir::GlobalOp>(name) &&
             "symbolTable and module out of sync");
#endif
      return global;
    }
  return modOp.lookupSymbol<fir::GlobalOp>(name);
}

aiir::Type fir::FirOpBuilder::getRefType(aiir::Type eleTy, bool isVolatile) {
  assert(!aiir::isa<fir::ReferenceType>(eleTy) && "cannot be a reference type");
  return fir::ReferenceType::get(eleTy, isVolatile);
}

aiir::Type fir::FirOpBuilder::getVarLenSeqTy(aiir::Type eleTy, unsigned rank) {
  fir::SequenceType::Shape shape(rank, fir::SequenceType::getUnknownExtent());
  return fir::SequenceType::get(shape, eleTy);
}

aiir::Type fir::FirOpBuilder::getRealType(int kind) {
  switch (kindMap.getRealTypeID(kind)) {
  case llvm::Type::TypeID::HalfTyID:
    return aiir::Float16Type::get(getContext());
  case llvm::Type::TypeID::BFloatTyID:
    return aiir::BFloat16Type::get(getContext());
  case llvm::Type::TypeID::FloatTyID:
    return aiir::Float32Type::get(getContext());
  case llvm::Type::TypeID::DoubleTyID:
    return aiir::Float64Type::get(getContext());
  case llvm::Type::TypeID::X86_FP80TyID:
    return aiir::Float80Type::get(getContext());
  case llvm::Type::TypeID::FP128TyID:
    return aiir::Float128Type::get(getContext());
  default:
    fir::emitFatalError(aiir::UnknownLoc::get(getContext()),
                        "unsupported type !fir.real<kind>");
  }
}

aiir::Value fir::FirOpBuilder::createNullConstant(aiir::Location loc,
                                                  aiir::Type ptrType) {
  auto ty = ptrType ? ptrType : getRefType(getNoneType());
  return fir::ZeroOp::create(*this, loc, ty);
}

aiir::Value fir::FirOpBuilder::createIntegerConstant(aiir::Location loc,
                                                     aiir::Type ty,
                                                     std::int64_t cst) {
  assert((cst >= 0 || aiir::isa<aiir::IndexType>(ty) ||
          aiir::cast<aiir::IntegerType>(ty).getWidth() <= 64) &&
         "must use APint");

  aiir::Type cstType = ty;
  if (auto intType = aiir::dyn_cast<aiir::IntegerType>(ty)) {
    // Signed and unsigned constants must be encoded as signless
    // arith.constant followed by fir.convert cast.
    if (intType.isUnsigned())
      cstType = aiir::IntegerType::get(getContext(), intType.getWidth());
    else if (intType.isSigned())
      TODO(loc, "signed integer constant");
  }

  aiir::Value cstValue = aiir::arith::ConstantOp::create(
      *this, loc, cstType, getIntegerAttr(cstType, cst));
  return createConvert(loc, ty, cstValue);
}

aiir::Value fir::FirOpBuilder::createAllOnesInteger(aiir::Location loc,
                                                    aiir::Type ty) {
  if (aiir::isa<aiir::IndexType>(ty))
    return createIntegerConstant(loc, ty, -1);
  llvm::APInt allOnes =
      llvm::APInt::getAllOnes(aiir::cast<aiir::IntegerType>(ty).getWidth());
  return aiir::arith::ConstantOp::create(*this, loc, ty,
                                         getIntegerAttr(ty, allOnes));
}

aiir::Value
fir::FirOpBuilder::createRealConstant(aiir::Location loc, aiir::Type fltTy,
                                      llvm::APFloat::integerPart val) {
  auto apf = [&]() -> llvm::APFloat {
    if (fltTy.isF16())
      return llvm::APFloat(llvm::APFloat::IEEEhalf(), val);
    if (fltTy.isBF16())
      return llvm::APFloat(llvm::APFloat::BFloat(), val);
    if (fltTy.isF32())
      return llvm::APFloat(llvm::APFloat::IEEEsingle(), val);
    if (fltTy.isF64())
      return llvm::APFloat(llvm::APFloat::IEEEdouble(), val);
    if (fltTy.isF80())
      return llvm::APFloat(llvm::APFloat::x87DoubleExtended(), val);
    if (fltTy.isF128())
      return llvm::APFloat(llvm::APFloat::IEEEquad(), val);
    llvm_unreachable("unhandled AIIR floating-point type");
  };
  return createRealConstant(loc, fltTy, apf());
}

aiir::Value fir::FirOpBuilder::createRealConstant(aiir::Location loc,
                                                  aiir::Type fltTy,
                                                  const llvm::APFloat &value) {
  if (aiir::isa<aiir::FloatType>(fltTy)) {
    auto attr = getFloatAttr(fltTy, value);
    return aiir::arith::ConstantOp::create(*this, loc, fltTy, attr);
  }
  llvm_unreachable("should use builtin floating-point type");
}

llvm::SmallVector<aiir::Value>
fir::factory::elideExtentsAlreadyInType(aiir::Type type,
                                        aiir::ValueRange shape) {
  auto arrTy = aiir::dyn_cast<fir::SequenceType>(type);
  if (shape.empty() || !arrTy)
    return {};
  // elide the constant dimensions before construction
  assert(shape.size() == arrTy.getDimension());
  llvm::SmallVector<aiir::Value> dynamicShape;
  auto typeShape = arrTy.getShape();
  for (unsigned i = 0, end = arrTy.getDimension(); i < end; ++i)
    if (typeShape[i] == fir::SequenceType::getUnknownExtent())
      dynamicShape.push_back(shape[i]);
  return dynamicShape;
}

llvm::SmallVector<aiir::Value>
fir::factory::elideLengthsAlreadyInType(aiir::Type type,
                                        aiir::ValueRange lenParams) {
  if (lenParams.empty())
    return {};
  if (auto arrTy = aiir::dyn_cast<fir::SequenceType>(type))
    type = arrTy.getEleTy();
  if (fir::hasDynamicSize(type))
    return lenParams;
  return {};
}

/// Allocate a local variable.
/// A local variable ought to have a name in the source code.
aiir::Value fir::FirOpBuilder::allocateLocal(
    aiir::Location loc, aiir::Type ty, llvm::StringRef uniqName,
    llvm::StringRef name, bool pinned, llvm::ArrayRef<aiir::Value> shape,
    llvm::ArrayRef<aiir::Value> lenParams, bool asTarget) {
  // Convert the shape extents to `index`, as needed.
  llvm::SmallVector<aiir::Value> indices;
  llvm::SmallVector<aiir::Value> elidedShape =
      fir::factory::elideExtentsAlreadyInType(ty, shape);
  llvm::SmallVector<aiir::Value> elidedLenParams =
      fir::factory::elideLengthsAlreadyInType(ty, lenParams);
  auto idxTy = getIndexType();
  for (aiir::Value sh : elidedShape)
    indices.push_back(createConvert(loc, idxTy, sh));
  // Add a target attribute, if needed.
  llvm::SmallVector<aiir::NamedAttribute> attrs;
  if (asTarget)
    attrs.emplace_back(
        aiir::StringAttr::get(getContext(), fir::getTargetAttrName()),
        getUnitAttr());
  // Create the local variable.
  if (name.empty()) {
    if (uniqName.empty())
      return fir::AllocaOp::create(*this, loc, ty, pinned, elidedLenParams,
                                   indices, attrs);
    return fir::AllocaOp::create(*this, loc, ty, uniqName, pinned,
                                 elidedLenParams, indices, attrs);
  }
  return fir::AllocaOp::create(*this, loc, ty, uniqName, name, pinned,
                               elidedLenParams, indices, attrs);
}

aiir::Value fir::FirOpBuilder::allocateLocal(
    aiir::Location loc, aiir::Type ty, llvm::StringRef uniqName,
    llvm::StringRef name, llvm::ArrayRef<aiir::Value> shape,
    llvm::ArrayRef<aiir::Value> lenParams, bool asTarget) {
  return allocateLocal(loc, ty, uniqName, name, /*pinned=*/false, shape,
                       lenParams, asTarget);
}

/// Get the block for adding Allocas.
aiir::Block *fir::FirOpBuilder::getAllocaBlock() {
  if (auto accComputeRegionIface =
          getRegion().getParentOfType<aiir::acc::ComputeRegionOpInterface>()) {
    return accComputeRegionIface.getAllocaBlock();
  }

  if (auto ompOutlineableIface =
          getRegion()
              .getParentOfType<aiir::omp::OutlineableOpenMPOpInterface>()) {
    return ompOutlineableIface.getAllocaBlock();
  }

  if (auto recipeIface =
          getRegion().getParentOfType<aiir::accomp::RecipeInterface>()) {
    return recipeIface.getAllocaBlock(getRegion());
  }

  if (auto cufKernelOp = getRegion().getParentOfType<cuf::KernelOp>())
    return &cufKernelOp.getRegion().front();

  if (auto doConcurentOp = getRegion().getParentOfType<fir::DoConcurrentOp>())
    return doConcurentOp.getBody();

  if (auto firLocalOp = getRegion().getParentOfType<fir::LocalitySpecifierOp>())
    return &getRegion().front();

  if (auto firLocalOp = getRegion().getParentOfType<fir::DeclareReductionOp>())
    return &getRegion().front();

  return getEntryBlock();
}

static aiir::ArrayAttr makeI64ArrayAttr(llvm::ArrayRef<int64_t> values,
                                        aiir::AIIRContext *context) {
  llvm::SmallVector<aiir::Attribute, 4> attrs;
  attrs.reserve(values.size());
  for (auto &v : values)
    attrs.push_back(aiir::IntegerAttr::get(aiir::IntegerType::get(context, 64),
                                           aiir::APInt(64, v)));
  return aiir::ArrayAttr::get(context, attrs);
}

aiir::ArrayAttr fir::FirOpBuilder::create2DI64ArrayAttr(
    llvm::SmallVectorImpl<llvm::SmallVector<int64_t>> &intData) {
  llvm::SmallVector<aiir::Attribute> arrayAttr;
  arrayAttr.reserve(intData.size());
  aiir::AIIRContext *context = getContext();
  for (auto &v : intData)
    arrayAttr.push_back(makeI64ArrayAttr(v, context));
  return aiir::ArrayAttr::get(context, arrayAttr);
}

aiir::Value fir::FirOpBuilder::createTemporaryAlloc(
    aiir::Location loc, aiir::Type type, llvm::StringRef name,
    aiir::ValueRange lenParams, aiir::ValueRange shape,
    llvm::ArrayRef<aiir::NamedAttribute> attrs,
    std::optional<Fortran::common::CUDADataAttr> cudaAttr) {
  assert(!aiir::isa<fir::ReferenceType>(type) && "cannot be a reference");
  // If the alloca is inside an OpenMP Op which will be outlined then pin
  // the alloca here.
  const bool pinned =
      getRegion().getParentOfType<aiir::omp::OutlineableOpenMPOpInterface>();
  if (cudaAttr) {
    cuf::DataAttributeAttr attr = cuf::getDataAttribute(getContext(), cudaAttr);
    return cuf::AllocOp::create(*this, loc, type,
                                /*unique_name=*/llvm::StringRef{}, name, attr,
                                lenParams, shape, attrs);
  } else {
    return fir::AllocaOp::create(*this, loc, type,
                                 /*unique_name=*/llvm::StringRef{}, name,
                                 pinned, lenParams, shape, attrs);
  }
}

/// Create a temporary variable on the stack. Anonymous temporaries have no
/// `name` value. Temporaries do not require a uniqued name.
aiir::Value fir::FirOpBuilder::createTemporary(
    aiir::Location loc, aiir::Type type, llvm::StringRef name,
    aiir::ValueRange shape, aiir::ValueRange lenParams,
    llvm::ArrayRef<aiir::NamedAttribute> attrs,
    std::optional<Fortran::common::CUDADataAttr> cudaAttr) {
  llvm::SmallVector<aiir::Value> dynamicShape =
      fir::factory::elideExtentsAlreadyInType(type, shape);
  llvm::SmallVector<aiir::Value> dynamicLength =
      fir::factory::elideLengthsAlreadyInType(type, lenParams);
  InsertPoint insPt;
  const bool hoistAlloc = dynamicShape.empty() && dynamicLength.empty();
  if (hoistAlloc) {
    insPt = saveInsertionPoint();
    setInsertionPointToStart(getAllocaBlock());
  }

  aiir::Value ae = createTemporaryAlloc(loc, type, name, dynamicLength,
                                        dynamicShape, attrs, cudaAttr);

  if (hoistAlloc)
    restoreInsertionPoint(insPt);
  return ae;
}

aiir::Value fir::FirOpBuilder::createHeapTemporary(
    aiir::Location loc, aiir::Type type, llvm::StringRef name,
    aiir::ValueRange shape, aiir::ValueRange lenParams,
    llvm::ArrayRef<aiir::NamedAttribute> attrs) {
  llvm::SmallVector<aiir::Value> dynamicShape =
      fir::factory::elideExtentsAlreadyInType(type, shape);
  llvm::SmallVector<aiir::Value> dynamicLength =
      fir::factory::elideLengthsAlreadyInType(type, lenParams);

  assert(!aiir::isa<fir::ReferenceType>(type) && "cannot be a reference");
  return fir::AllocMemOp::create(*this, loc, type,
                                 /*unique_name=*/llvm::StringRef{}, name,
                                 dynamicLength, dynamicShape, attrs);
}

std::pair<aiir::Value, bool> fir::FirOpBuilder::createAndDeclareTemp(
    aiir::Location loc, aiir::Type baseType, aiir::Value shape,
    llvm::ArrayRef<aiir::Value> extents, llvm::ArrayRef<aiir::Value> typeParams,
    const std::function<decltype(FirOpBuilder::genTempDeclareOp)> &genDeclare,
    aiir::Value polymorphicMold, bool useStack, llvm::StringRef tmpName) {
  if (polymorphicMold) {
    // Create *allocated* polymorphic temporary using the dynamic type
    // of the mold and the provided shape/extents.
    auto boxType = fir::ClassType::get(fir::HeapType::get(baseType));
    aiir::Value boxAddress = fir::factory::getAndEstablishBoxStorage(
        *this, loc, boxType, shape, typeParams, polymorphicMold);
    fir::runtime::genAllocatableAllocate(*this, loc, boxAddress);
    aiir::Value box = fir::LoadOp::create(*this, loc, boxAddress);
    aiir::Value base =
        genDeclare(*this, loc, box, tmpName, /*shape=*/aiir::Value{},
                   typeParams, fir::FortranVariableFlagsAttr{});
    return {base, /*isHeapAllocation=*/true};
  }
  aiir::Value allocmem;
  if (useStack)
    allocmem = createTemporary(loc, baseType, tmpName, extents, typeParams);
  else
    allocmem = createHeapTemporary(loc, baseType, tmpName, extents, typeParams);
  aiir::Value base = genDeclare(*this, loc, allocmem, tmpName, shape,
                                typeParams, fir::FortranVariableFlagsAttr{});
  return {base, !useStack};
}

aiir::Value fir::FirOpBuilder::genTempDeclareOp(
    fir::FirOpBuilder &builder, aiir::Location loc, aiir::Value memref,
    llvm::StringRef name, aiir::Value shape,
    llvm::ArrayRef<aiir::Value> typeParams,
    fir::FortranVariableFlagsAttr fortranAttrs) {
  auto nameAttr = aiir::StringAttr::get(builder.getContext(), name);
  return fir::DeclareOp::create(
      builder, loc, memref.getType(), memref, shape, typeParams,
      /*dummy_scope=*/nullptr,
      /*storage=*/nullptr,
      /*storage_offset=*/0, nameAttr, fortranAttrs, cuf::DataAttributeAttr{},
      /*dummy_arg_no=*/aiir::IntegerAttr{});
}

aiir::Value fir::FirOpBuilder::genStackSave(aiir::Location loc) {
  aiir::Type voidPtr = aiir::LLVM::LLVMPointerType::get(
      getContext(), fir::factory::getAllocaAddressSpace(&getDataLayout()));
  return aiir::LLVM::StackSaveOp::create(*this, loc, voidPtr);
}

void fir::FirOpBuilder::genStackRestore(aiir::Location loc,
                                        aiir::Value stackPointer) {
  aiir::LLVM::StackRestoreOp::create(*this, loc, stackPointer);
}

/// Create a global variable in the (read-only) data section. A global variable
/// must have a unique name to identify and reference it.
fir::GlobalOp fir::FirOpBuilder::createGlobal(
    aiir::Location loc, aiir::Type type, llvm::StringRef name,
    aiir::StringAttr linkage, aiir::Attribute value, bool isConst,
    bool isTarget, cuf::DataAttributeAttr dataAttr) {
  if (auto global = getNamedGlobal(name))
    return global;
  auto module = getModule();
  auto insertPt = saveInsertionPoint();
  setInsertionPoint(module.getBody(), module.getBody()->end());
  llvm::SmallVector<aiir::NamedAttribute> attrs;
  if (dataAttr) {
    auto globalOpName = aiir::OperationName(fir::GlobalOp::getOperationName(),
                                            module.getContext());
    attrs.push_back(aiir::NamedAttribute(
        fir::GlobalOp::getDataAttrAttrName(globalOpName), dataAttr));
  }
  auto glob = fir::GlobalOp::create(*this, loc, name, isConst, isTarget, type,
                                    value, linkage, attrs);
  restoreInsertionPoint(insertPt);
  if (symbolTable)
    symbolTable->insert(glob);
  return glob;
}

fir::GlobalOp fir::FirOpBuilder::createGlobal(
    aiir::Location loc, aiir::Type type, llvm::StringRef name, bool isConst,
    bool isTarget, std::function<void(FirOpBuilder &)> bodyBuilder,
    aiir::StringAttr linkage, cuf::DataAttributeAttr dataAttr) {
  if (auto global = getNamedGlobal(name))
    return global;
  auto module = getModule();
  auto insertPt = saveInsertionPoint();
  setInsertionPoint(module.getBody(), module.getBody()->end());
  auto glob = fir::GlobalOp::create(*this, loc, name, isConst, isTarget, type,
                                    aiir::Attribute{}, linkage);
  auto &region = glob.getRegion();
  region.push_back(new aiir::Block);
  auto &block = glob.getRegion().back();
  setInsertionPointToStart(&block);
  bodyBuilder(*this);
  restoreInsertionPoint(insertPt);
  if (symbolTable)
    symbolTable->insert(glob);
  return glob;
}

std::pair<fir::TypeInfoOp, aiir::OpBuilder::InsertPoint>
fir::FirOpBuilder::createTypeInfoOp(aiir::Location loc,
                                    fir::RecordType recordType,
                                    fir::RecordType parentType) {
  aiir::ModuleOp module = getModule();
  if (fir::TypeInfoOp typeInfo =
          fir::lookupTypeInfoOp(recordType.getName(), module, symbolTable))
    return {typeInfo, InsertPoint{}};
  InsertPoint insertPoint = saveInsertionPoint();
  setInsertionPoint(module.getBody(), module.getBody()->end());
  auto typeInfo = fir::TypeInfoOp::create(*this, loc, recordType, parentType);
  if (symbolTable)
    symbolTable->insert(typeInfo);
  return {typeInfo, insertPoint};
}

aiir::Value fir::FirOpBuilder::convertWithSemantics(
    aiir::Location loc, aiir::Type toTy, aiir::Value val,
    bool allowCharacterConversion, bool allowRebox) {
  assert(toTy && "store location must be typed");
  auto fromTy = val.getType();
  if (fromTy == toTy)
    return val;
  fir::factory::Complex helper{*this, loc};
  if ((fir::isa_real(fromTy) || fir::isa_integer(fromTy)) &&
      fir::isa_complex(toTy)) {
    // imaginary part is zero
    auto eleTy = helper.getComplexPartType(toTy);
    auto cast = createConvert(loc, eleTy, val);
    auto imag = createRealZeroConstant(loc, eleTy);
    return helper.createComplex(toTy, cast, imag);
  }
  if (fir::isa_complex(fromTy) &&
      (fir::isa_integer(toTy) || fir::isa_real(toTy))) {
    // drop the imaginary part
    auto rp = helper.extractComplexPart(val, /*isImagPart=*/false);
    return createConvert(loc, toTy, rp);
  }
  if (allowCharacterConversion) {
    if (aiir::isa<fir::BoxCharType>(fromTy)) {
      // Extract the address of the character string and pass it
      fir::factory::CharacterExprHelper charHelper{*this, loc};
      std::pair<aiir::Value, aiir::Value> unboxchar =
          charHelper.createUnboxChar(val);
      return createConvert(loc, toTy, unboxchar.first);
    }
    if (auto boxType = aiir::dyn_cast<fir::BoxCharType>(toTy)) {
      // Extract the address of the actual argument and create a boxed
      // character value with an undefined length
      // TODO: We should really calculate the total size of the actual
      // argument in characters and use it as the length of the string
      auto refType = getRefType(boxType.getEleTy());
      aiir::Value charBase = createConvert(loc, refType, val);
      // Do not use fir.undef since llvm optimizer is too harsh when it
      // sees such values (may just delete code).
      aiir::Value unknownLen = createIntegerConstant(loc, getIndexType(), 0);
      fir::factory::CharacterExprHelper charHelper{*this, loc};
      return charHelper.createEmboxChar(charBase, unknownLen);
    }
  }
  if (fir::isa_ref_type(toTy) && fir::isa_box_type(fromTy)) {
    // Call is expecting a raw data pointer, not a box. Get the data pointer out
    // of the box and pass that.
    assert((fir::unwrapRefType(toTy) ==
                fir::unwrapRefType(fir::unwrapPassByRefType(fromTy)) &&
            "element types expected to match"));
    return fir::BoxAddrOp::create(*this, loc, toTy, val);
  }
  if (fir::isa_ref_type(fromTy) && aiir::isa<fir::BoxProcType>(toTy)) {
    // Call is expecting a boxed procedure, not a reference to other data type.
    // Convert the reference to a procedure and embox it.
    aiir::Type procTy = aiir::cast<fir::BoxProcType>(toTy).getEleTy();
    aiir::Value proc = createConvert(loc, procTy, val);
    return fir::EmboxProcOp::create(*this, loc, toTy, proc);
  }

  // Legacy: remove when removing non HLFIR lowering path.
  if (allowRebox)
    if (((fir::isPolymorphicType(fromTy) &&
          (fir::isAllocatableType(fromTy) || fir::isPointerType(fromTy)) &&
          fir::isPolymorphicType(toTy)) ||
         (fir::isPolymorphicType(fromTy) && aiir::isa<fir::BoxType>(toTy))) &&
        !(fir::isUnlimitedPolymorphicType(fromTy) && fir::isAssumedType(toTy)))
      return fir::ReboxOp::create(*this, loc, toTy, val, aiir::Value{},
                                  /*slice=*/aiir::Value{});

  return createConvert(loc, toTy, val);
}

aiir::Value fir::FirOpBuilder::createVolatileCast(aiir::Location loc,
                                                  bool isVolatile,
                                                  aiir::Value val) {
  aiir::Type volatileAdjustedType =
      fir::updateTypeWithVolatility(val.getType(), isVolatile);
  if (volatileAdjustedType == val.getType())
    return val;
  return fir::VolatileCastOp::create(*this, loc, volatileAdjustedType, val);
}

aiir::Value fir::FirOpBuilder::createConvertWithVolatileCast(aiir::Location loc,
                                                             aiir::Type toTy,
                                                             aiir::Value val) {
  val = createVolatileCast(loc, fir::isa_volatile_type(toTy), val);
  return createConvert(loc, toTy, val);
}

aiir::Value fir::factory::createConvert(aiir::OpBuilder &builder,
                                        aiir::Location loc, aiir::Type toTy,
                                        aiir::Value val) {
  if (val.getType() != toTy) {
    assert((!fir::isa_derived(toTy) ||
            aiir::cast<fir::RecordType>(val.getType()).getTypeList() ==
                aiir::cast<fir::RecordType>(toTy).getTypeList()) &&
           "incompatible record types");
    return fir::ConvertOp::create(builder, loc, toTy, val);
  }
  return val;
}

aiir::Value fir::FirOpBuilder::createConvert(aiir::Location loc,
                                             aiir::Type toTy, aiir::Value val) {
  return fir::factory::createConvert(*this, loc, toTy, val);
}

void fir::FirOpBuilder::createStoreWithConvert(aiir::Location loc,
                                               aiir::Value val,
                                               aiir::Value addr) {
  aiir::Type unwrapedRefType = fir::unwrapRefType(addr.getType());
  val = createVolatileCast(loc, fir::isa_volatile_type(unwrapedRefType), val);
  aiir::Value cast = createConvert(loc, unwrapedRefType, val);
  fir::StoreOp::create(*this, loc, cast, addr);
}

aiir::Value fir::FirOpBuilder::loadIfRef(aiir::Location loc, aiir::Value val) {
  if (fir::isa_ref_type(val.getType()))
    return fir::LoadOp::create(*this, loc, val);
  return val;
}

fir::StringLitOp fir::FirOpBuilder::createStringLitOp(aiir::Location loc,
                                                      llvm::StringRef data) {
  auto type = fir::CharacterType::get(getContext(), 1, data.size());
  auto strAttr = aiir::StringAttr::get(getContext(), data);
  auto valTag = aiir::StringAttr::get(getContext(), fir::StringLitOp::value());
  aiir::NamedAttribute dataAttr(valTag, strAttr);
  auto sizeTag = aiir::StringAttr::get(getContext(), fir::StringLitOp::size());
  aiir::NamedAttribute sizeAttr(sizeTag, getI64IntegerAttr(data.size()));
  llvm::SmallVector<aiir::NamedAttribute> attrs{dataAttr, sizeAttr};
  return fir::StringLitOp::create(*this, loc, llvm::ArrayRef<aiir::Type>{type},
                                  aiir::ValueRange{}, attrs);
}

aiir::Value fir::FirOpBuilder::genShape(aiir::Location loc,
                                        llvm::ArrayRef<aiir::Value> exts) {
  return fir::ShapeOp::create(*this, loc, exts);
}

aiir::Value fir::FirOpBuilder::genShape(aiir::Location loc,
                                        llvm::ArrayRef<aiir::Value> shift,
                                        llvm::ArrayRef<aiir::Value> exts) {
  auto shapeType = fir::ShapeShiftType::get(getContext(), exts.size());
  llvm::SmallVector<aiir::Value> shapeArgs;
  auto idxTy = getIndexType();
  for (auto [lbnd, ext] : llvm::zip(shift, exts)) {
    auto lb = createConvert(loc, idxTy, lbnd);
    shapeArgs.push_back(lb);
    shapeArgs.push_back(ext);
  }
  return fir::ShapeShiftOp::create(*this, loc, shapeType, shapeArgs);
}

aiir::Value fir::FirOpBuilder::genShape(aiir::Location loc,
                                        const fir::AbstractArrayBox &arr) {
  if (arr.lboundsAllOne())
    return genShape(loc, arr.getExtents());
  return genShape(loc, arr.getLBounds(), arr.getExtents());
}

aiir::Value fir::FirOpBuilder::genShift(aiir::Location loc,
                                        llvm::ArrayRef<aiir::Value> shift) {
  auto shiftType = fir::ShiftType::get(getContext(), shift.size());
  return fir::ShiftOp::create(*this, loc, shiftType, shift);
}

aiir::Value fir::FirOpBuilder::createShape(aiir::Location loc,
                                           const fir::ExtendedValue &exv) {
  return exv.match(
      [&](const fir::ArrayBoxValue &box) { return genShape(loc, box); },
      [&](const fir::CharArrayBoxValue &box) { return genShape(loc, box); },
      [&](const fir::BoxValue &box) -> aiir::Value {
        if (!box.getLBounds().empty()) {
          auto shiftType =
              fir::ShiftType::get(getContext(), box.getLBounds().size());
          return fir::ShiftOp::create(*this, loc, shiftType, box.getLBounds());
        }
        return {};
      },
      [&](const fir::MutableBoxValue &) -> aiir::Value {
        // MutableBoxValue must be read into another category to work with them
        // outside of allocation/assignment contexts.
        fir::emitFatalError(loc, "createShape on MutableBoxValue");
      },
      [&](auto) -> aiir::Value { fir::emitFatalError(loc, "not an array"); });
}

aiir::Value fir::FirOpBuilder::createSlice(aiir::Location loc,
                                           const fir::ExtendedValue &exv,
                                           aiir::ValueRange triples,
                                           aiir::ValueRange path) {
  if (triples.empty()) {
    // If there is no slicing by triple notation, then take the whole array.
    auto fullShape = [&](const llvm::ArrayRef<aiir::Value> lbounds,
                         llvm::ArrayRef<aiir::Value> extents) -> aiir::Value {
      llvm::SmallVector<aiir::Value> trips;
      auto idxTy = getIndexType();
      auto one = createIntegerConstant(loc, idxTy, 1);
      if (lbounds.empty()) {
        for (auto v : extents) {
          trips.push_back(one);
          trips.push_back(v);
          trips.push_back(one);
        }
        return fir::SliceOp::create(*this, loc, trips, path);
      }
      for (auto [lbnd, extent] : llvm::zip(lbounds, extents)) {
        auto lb = createConvert(loc, idxTy, lbnd);
        auto ext = createConvert(loc, idxTy, extent);
        auto shift = aiir::arith::SubIOp::create(*this, loc, lb, one);
        auto ub = aiir::arith::AddIOp::create(*this, loc, ext, shift);
        trips.push_back(lb);
        trips.push_back(ub);
        trips.push_back(one);
      }
      return fir::SliceOp::create(*this, loc, trips, path);
    };
    return exv.match(
        [&](const fir::ArrayBoxValue &box) {
          return fullShape(box.getLBounds(), box.getExtents());
        },
        [&](const fir::CharArrayBoxValue &box) {
          return fullShape(box.getLBounds(), box.getExtents());
        },
        [&](const fir::BoxValue &box) {
          auto extents = fir::factory::readExtents(*this, loc, box);
          return fullShape(box.getLBounds(), extents);
        },
        [&](const fir::MutableBoxValue &) -> aiir::Value {
          // MutableBoxValue must be read into another category to work with
          // them outside of allocation/assignment contexts.
          fir::emitFatalError(loc, "createSlice on MutableBoxValue");
        },
        [&](auto) -> aiir::Value { fir::emitFatalError(loc, "not an array"); });
  }
  return fir::SliceOp::create(*this, loc, triples, path);
}

aiir::Value fir::FirOpBuilder::createBox(aiir::Location loc,
                                         const fir::ExtendedValue &exv,
                                         bool isPolymorphic,
                                         bool isAssumedType) {
  aiir::Value itemAddr = fir::getBase(exv);
  if (aiir::isa<fir::BaseBoxType>(itemAddr.getType()))
    return itemAddr;
  auto elementType = fir::dyn_cast_ptrEleTy(itemAddr.getType());
  if (!elementType) {
    aiir::emitError(loc, "internal: expected a memory reference type ")
        << itemAddr.getType();
    llvm_unreachable("not a memory reference type");
  }
  const bool isVolatile = fir::isa_volatile_type(itemAddr.getType());
  aiir::Type boxTy;
  aiir::Value tdesc;
  // Avoid to wrap a box/class with box/class.
  if (aiir::isa<fir::BaseBoxType>(elementType)) {
    boxTy = elementType;
  } else {
    boxTy = fir::BoxType::get(elementType, isVolatile);
    if (isPolymorphic) {
      elementType = fir::updateTypeForUnlimitedPolymorphic(elementType);
      if (isAssumedType)
        boxTy = fir::BoxType::get(elementType, isVolatile);
      else
        boxTy = fir::ClassType::get(elementType, isVolatile);
    }
  }

  return exv.match(
      [&](const fir::ArrayBoxValue &box) -> aiir::Value {
        aiir::Value empty;
        aiir::ValueRange emptyRange;
        aiir::Value s = createShape(loc, exv);
        return fir::EmboxOp::create(*this, loc, boxTy, itemAddr, s,
                                    /*slice=*/empty,
                                    /*typeparams=*/emptyRange,
                                    isPolymorphic ? box.getSourceBox() : tdesc);
      },
      [&](const fir::CharArrayBoxValue &box) -> aiir::Value {
        aiir::Value s = createShape(loc, exv);
        if (fir::factory::CharacterExprHelper::hasConstantLengthInType(exv))
          return fir::EmboxOp::create(*this, loc, boxTy, itemAddr, s);

        aiir::Value emptySlice;
        llvm::SmallVector<aiir::Value> lenParams{box.getLen()};
        return fir::EmboxOp::create(*this, loc, boxTy, itemAddr, s, emptySlice,
                                    lenParams);
      },
      [&](const fir::CharBoxValue &box) -> aiir::Value {
        if (fir::factory::CharacterExprHelper::hasConstantLengthInType(exv))
          return fir::EmboxOp::create(*this, loc, boxTy, itemAddr);
        aiir::Value emptyShape, emptySlice;
        llvm::SmallVector<aiir::Value> lenParams{box.getLen()};
        return fir::EmboxOp::create(*this, loc, boxTy, itemAddr, emptyShape,
                                    emptySlice, lenParams);
      },
      [&](const fir::MutableBoxValue &x) -> aiir::Value {
        return fir::LoadOp::create(
            *this, loc, fir::factory::getMutableIRBox(*this, loc, x));
      },
      [&](const fir::PolymorphicValue &p) -> aiir::Value {
        aiir::Value empty;
        aiir::ValueRange emptyRange;
        return fir::EmboxOp::create(*this, loc, boxTy, itemAddr, empty, empty,
                                    emptyRange,
                                    isPolymorphic ? p.getSourceBox() : tdesc);
      },
      [&](const auto &) -> aiir::Value {
        aiir::Value empty;
        aiir::ValueRange emptyRange;
        return fir::EmboxOp::create(*this, loc, boxTy, itemAddr, empty, empty,
                                    emptyRange, tdesc);
      });
}

aiir::Value fir::FirOpBuilder::createBox(aiir::Location loc, aiir::Type boxType,
                                         aiir::Value addr, aiir::Value shape,
                                         aiir::Value slice,
                                         llvm::ArrayRef<aiir::Value> lengths,
                                         aiir::Value tdesc) {
  aiir::Type valueOrSequenceType = fir::unwrapPassByRefType(boxType);
  return fir::EmboxOp::create(
      *this, loc, boxType, addr, shape, slice,
      fir::factory::elideLengthsAlreadyInType(valueOrSequenceType, lengths),
      tdesc);
}

void fir::FirOpBuilder::dumpFunc() { getFunction().dump(); }

static aiir::Value
genNullPointerComparison(fir::FirOpBuilder &builder, aiir::Location loc,
                         aiir::Value addr,
                         aiir::arith::CmpIPredicate condition) {
  auto intPtrTy = builder.getIntPtrType();
  auto ptrToInt = builder.createConvert(loc, intPtrTy, addr);
  auto c0 = builder.createIntegerConstant(loc, intPtrTy, 0);
  return aiir::arith::CmpIOp::create(builder, loc, condition, ptrToInt, c0);
}

aiir::Value fir::FirOpBuilder::genIsNotNullAddr(aiir::Location loc,
                                                aiir::Value addr) {
  return genNullPointerComparison(*this, loc, addr,
                                  aiir::arith::CmpIPredicate::ne);
}

aiir::Value fir::FirOpBuilder::genIsNullAddr(aiir::Location loc,
                                             aiir::Value addr) {
  return genNullPointerComparison(*this, loc, addr,
                                  aiir::arith::CmpIPredicate::eq);
}

template <typename OpTy, typename... Args>
static aiir::Value createAndMaybeFold(bool fold, fir::FirOpBuilder &builder,
                                      aiir::Location loc, Args &&...args) {
  if (fold)
    return builder.createOrFold<OpTy>(loc, std::forward<Args>(args)...);
  return OpTy::create(builder, loc, std::forward<Args>(args)...);
}

aiir::Value
fir::FirOpBuilder::genExtentFromTriplet(aiir::Location loc, aiir::Value lb,
                                        aiir::Value ub, aiir::Value step,
                                        aiir::Type type, bool fold) {
  auto zero = createIntegerConstant(loc, type, 0);
  lb = createConvert(loc, type, lb);
  ub = createConvert(loc, type, ub);
  step = createConvert(loc, type, step);

  auto diff = createAndMaybeFold<aiir::arith::SubIOp>(fold, *this, loc, ub, lb);
  auto add =
      createAndMaybeFold<aiir::arith::AddIOp>(fold, *this, loc, diff, step);
  auto div =
      createAndMaybeFold<aiir::arith::DivSIOp>(fold, *this, loc, add, step);
  auto cmp = createAndMaybeFold<aiir::arith::CmpIOp>(
      fold, *this, loc, aiir::arith::CmpIPredicate::sgt, div, zero);
  return createAndMaybeFold<aiir::arith::SelectOp>(fold, *this, loc, cmp, div,
                                                   zero);
}

aiir::Value fir::FirOpBuilder::genAbsentOp(aiir::Location loc,
                                           aiir::Type argTy) {
  if (!fir::isCharacterProcedureTuple(argTy))
    return fir::AbsentOp::create(*this, loc, argTy);

  auto boxProc = fir::AbsentOp::create(
      *this, loc, aiir::cast<aiir::TupleType>(argTy).getType(0));
  aiir::Value charLen =
      fir::UndefOp::create(*this, loc, getCharacterLengthType());
  return fir::factory::createCharacterProcedureTuple(*this, loc, argTy, boxProc,
                                                     charLen);
}

void fir::FirOpBuilder::setCommonAttributes(aiir::Operation *op) const {
  auto fmi = aiir::dyn_cast<aiir::arith::ArithFastMathInterface>(*op);
  if (fmi) {
    // TODO: use fmi.setFastMathFlagsAttr() after D137114 is merged.
    //       For now set the attribute by the name.
    llvm::StringRef arithFMFAttrName = fmi.getFastMathAttrName();
    if (fastMathFlags != aiir::arith::FastMathFlags::none)
      op->setAttr(arithFMFAttrName, aiir::arith::FastMathFlagsAttr::get(
                                        op->getContext(), fastMathFlags));
  }
  auto iofi =
      aiir::dyn_cast<aiir::arith::ArithIntegerOverflowFlagsInterface>(*op);
  if (iofi) {
    llvm::StringRef arithIOFAttrName = iofi.getIntegerOverflowAttrName();
    if (integerOverflowFlags != aiir::arith::IntegerOverflowFlags::none)
      op->setAttr(arithIOFAttrName,
                  aiir::arith::IntegerOverflowFlagsAttr::get(
                      op->getContext(), integerOverflowFlags));
  }
}

void fir::FirOpBuilder::setFastMathFlags(
    Fortran::common::MathOptionsBase options) {
  aiir::arith::FastMathFlags arithFMF{};
  if (options.getFPContractEnabled()) {
    arithFMF = arithFMF | aiir::arith::FastMathFlags::contract;
  }
  if (options.getNoHonorInfs()) {
    arithFMF = arithFMF | aiir::arith::FastMathFlags::ninf;
  }
  if (options.getNoHonorNaNs()) {
    arithFMF = arithFMF | aiir::arith::FastMathFlags::nnan;
  }
  if (options.getApproxFunc()) {
    arithFMF = arithFMF | aiir::arith::FastMathFlags::afn;
  }
  if (options.getNoSignedZeros()) {
    arithFMF = arithFMF | aiir::arith::FastMathFlags::nsz;
  }
  if (options.getAssociativeMath()) {
    arithFMF = arithFMF | aiir::arith::FastMathFlags::reassoc;
  }
  if (options.getReciprocalMath()) {
    arithFMF = arithFMF | aiir::arith::FastMathFlags::arcp;
  }
  setFastMathFlags(arithFMF);
}

// Construction of an aiir::DataLayout is expensive so only do it on demand and
// memoise it in the builder instance
aiir::DataLayout &fir::FirOpBuilder::getDataLayout() {
  if (dataLayout)
    return *dataLayout;
  dataLayout = std::make_unique<aiir::DataLayout>(getModule());
  return *dataLayout;
}

//===--------------------------------------------------------------------===//
// ExtendedValue inquiry helper implementation
//===--------------------------------------------------------------------===//

aiir::Value fir::factory::readCharLen(fir::FirOpBuilder &builder,
                                      aiir::Location loc,
                                      const fir::ExtendedValue &box) {
  return box.match(
      [&](const fir::CharBoxValue &x) -> aiir::Value { return x.getLen(); },
      [&](const fir::CharArrayBoxValue &x) -> aiir::Value {
        return x.getLen();
      },
      [&](const fir::BoxValue &x) -> aiir::Value {
        assert(x.isCharacter());
        if (!x.getExplicitParameters().empty())
          return x.getExplicitParameters()[0];
        return fir::factory::CharacterExprHelper{builder, loc}
            .readLengthFromBox(x.getAddr());
      },
      [&](const fir::MutableBoxValue &x) -> aiir::Value {
        return readCharLen(builder, loc,
                           fir::factory::genMutableBoxRead(builder, loc, x));
      },
      [&](const auto &) -> aiir::Value {
        fir::emitFatalError(
            loc, "Character length inquiry on a non-character entity");
      });
}

aiir::Value fir::factory::readExtent(fir::FirOpBuilder &builder,
                                     aiir::Location loc,
                                     const fir::ExtendedValue &box,
                                     unsigned dim) {
  assert(box.rank() > dim);
  return box.match(
      [&](const fir::ArrayBoxValue &x) -> aiir::Value {
        return x.getExtents()[dim];
      },
      [&](const fir::CharArrayBoxValue &x) -> aiir::Value {
        return x.getExtents()[dim];
      },
      [&](const fir::BoxValue &x) -> aiir::Value {
        if (!x.getExplicitExtents().empty())
          return x.getExplicitExtents()[dim];
        auto idxTy = builder.getIndexType();
        auto dimVal = builder.createIntegerConstant(loc, idxTy, dim);
        return fir::BoxDimsOp::create(builder, loc, idxTy, idxTy, idxTy,
                                      x.getAddr(), dimVal)
            .getResult(1);
      },
      [&](const fir::MutableBoxValue &x) -> aiir::Value {
        return readExtent(builder, loc,
                          fir::factory::genMutableBoxRead(builder, loc, x),
                          dim);
      },
      [&](const auto &) -> aiir::Value {
        fir::emitFatalError(loc, "extent inquiry on scalar");
      });
}

aiir::Value fir::factory::readLowerBound(fir::FirOpBuilder &builder,
                                         aiir::Location loc,
                                         const fir::ExtendedValue &box,
                                         unsigned dim,
                                         aiir::Value defaultValue) {
  assert(box.rank() > dim);
  auto lb = box.match(
      [&](const fir::ArrayBoxValue &x) -> aiir::Value {
        return x.getLBounds().empty() ? aiir::Value{} : x.getLBounds()[dim];
      },
      [&](const fir::CharArrayBoxValue &x) -> aiir::Value {
        return x.getLBounds().empty() ? aiir::Value{} : x.getLBounds()[dim];
      },
      [&](const fir::BoxValue &x) -> aiir::Value {
        return x.getLBounds().empty() ? aiir::Value{} : x.getLBounds()[dim];
      },
      [&](const fir::MutableBoxValue &x) -> aiir::Value {
        return readLowerBound(builder, loc,
                              fir::factory::genMutableBoxRead(builder, loc, x),
                              dim, defaultValue);
      },
      [&](const auto &) -> aiir::Value {
        fir::emitFatalError(loc, "lower bound inquiry on scalar");
      });
  if (lb)
    return lb;
  return defaultValue;
}

llvm::SmallVector<aiir::Value>
fir::factory::readExtents(fir::FirOpBuilder &builder, aiir::Location loc,
                          const fir::BoxValue &box) {
  llvm::SmallVector<aiir::Value> result;
  auto explicitExtents = box.getExplicitExtents();
  if (!explicitExtents.empty()) {
    result.append(explicitExtents.begin(), explicitExtents.end());
    return result;
  }
  auto rank = box.rank();
  auto idxTy = builder.getIndexType();
  for (decltype(rank) dim = 0; dim < rank; ++dim) {
    auto dimVal = builder.createIntegerConstant(loc, idxTy, dim);
    auto dimInfo = fir::BoxDimsOp::create(builder, loc, idxTy, idxTy, idxTy,
                                          box.getAddr(), dimVal);
    result.emplace_back(dimInfo.getResult(1));
  }
  return result;
}

llvm::SmallVector<aiir::Value>
fir::factory::getExtents(aiir::Location loc, fir::FirOpBuilder &builder,
                         const fir::ExtendedValue &box) {
  return box.match(
      [&](const fir::ArrayBoxValue &x) -> llvm::SmallVector<aiir::Value> {
        return {x.getExtents().begin(), x.getExtents().end()};
      },
      [&](const fir::CharArrayBoxValue &x) -> llvm::SmallVector<aiir::Value> {
        return {x.getExtents().begin(), x.getExtents().end()};
      },
      [&](const fir::BoxValue &x) -> llvm::SmallVector<aiir::Value> {
        return fir::factory::readExtents(builder, loc, x);
      },
      [&](const fir::MutableBoxValue &x) -> llvm::SmallVector<aiir::Value> {
        auto load = fir::factory::genMutableBoxRead(builder, loc, x);
        return fir::factory::getExtents(loc, builder, load);
      },
      [&](const auto &) -> llvm::SmallVector<aiir::Value> { return {}; });
}

fir::ExtendedValue fir::factory::readBoxValue(fir::FirOpBuilder &builder,
                                              aiir::Location loc,
                                              const fir::BoxValue &box) {
  assert(!box.hasAssumedRank() &&
         "cannot read unlimited polymorphic or assumed rank fir.box");
  auto addr =
      fir::BoxAddrOp::create(builder, loc, box.getMemTy(), box.getAddr());
  if (box.isCharacter()) {
    auto len = fir::factory::readCharLen(builder, loc, box);
    if (box.rank() == 0)
      return fir::CharBoxValue(addr, len);
    return fir::CharArrayBoxValue(addr, len,
                                  fir::factory::readExtents(builder, loc, box),
                                  box.getLBounds());
  }
  if (box.isDerivedWithLenParameters())
    TODO(loc, "read fir.box with length parameters");
  aiir::Value sourceBox;
  if (box.isPolymorphic())
    sourceBox = box.getAddr();
  if (box.isPolymorphic() && box.rank() == 0)
    return fir::PolymorphicValue(addr, sourceBox);
  if (box.rank() == 0)
    return addr;
  return fir::ArrayBoxValue(addr, fir::factory::readExtents(builder, loc, box),
                            box.getLBounds(), sourceBox);
}

llvm::SmallVector<aiir::Value>
fir::factory::getNonDefaultLowerBounds(fir::FirOpBuilder &builder,
                                       aiir::Location loc,
                                       const fir::ExtendedValue &exv) {
  return exv.match(
      [&](const fir::ArrayBoxValue &array) -> llvm::SmallVector<aiir::Value> {
        return {array.getLBounds().begin(), array.getLBounds().end()};
      },
      [&](const fir::CharArrayBoxValue &array)
          -> llvm::SmallVector<aiir::Value> {
        return {array.getLBounds().begin(), array.getLBounds().end()};
      },
      [&](const fir::BoxValue &box) -> llvm::SmallVector<aiir::Value> {
        return {box.getLBounds().begin(), box.getLBounds().end()};
      },
      [&](const fir::MutableBoxValue &box) -> llvm::SmallVector<aiir::Value> {
        auto load = fir::factory::genMutableBoxRead(builder, loc, box);
        return fir::factory::getNonDefaultLowerBounds(builder, loc, load);
      },
      [&](const auto &) -> llvm::SmallVector<aiir::Value> { return {}; });
}

llvm::SmallVector<aiir::Value>
fir::factory::getNonDeferredLenParams(const fir::ExtendedValue &exv) {
  return exv.match(
      [&](const fir::CharArrayBoxValue &character)
          -> llvm::SmallVector<aiir::Value> { return {character.getLen()}; },
      [&](const fir::CharBoxValue &character)
          -> llvm::SmallVector<aiir::Value> { return {character.getLen()}; },
      [&](const fir::MutableBoxValue &box) -> llvm::SmallVector<aiir::Value> {
        return {box.nonDeferredLenParams().begin(),
                box.nonDeferredLenParams().end()};
      },
      [&](const fir::BoxValue &box) -> llvm::SmallVector<aiir::Value> {
        return {box.getExplicitParameters().begin(),
                box.getExplicitParameters().end()};
      },
      [&](const auto &) -> llvm::SmallVector<aiir::Value> { return {}; });
}

// If valTy is a box type, then we need to extract the type parameters from
// the box value.
static llvm::SmallVector<aiir::Value> getFromBox(aiir::Location loc,
                                                 fir::FirOpBuilder &builder,
                                                 aiir::Type valTy,
                                                 aiir::Value boxVal) {
  if (auto boxTy = aiir::dyn_cast<fir::BaseBoxType>(valTy)) {
    auto eleTy = fir::unwrapAllRefAndSeqType(boxTy.getEleTy());
    if (auto recTy = aiir::dyn_cast<fir::RecordType>(eleTy)) {
      if (recTy.getNumLenParams() > 0) {
        // Walk each type parameter in the record and get the value.
        TODO(loc, "generate code to get LEN type parameters");
      }
    } else if (auto charTy = aiir::dyn_cast<fir::CharacterType>(eleTy)) {
      if (charTy.hasDynamicLen()) {
        auto idxTy = builder.getIndexType();
        auto eleSz = fir::BoxEleSizeOp::create(builder, loc, idxTy, boxVal);
        auto kindBytes =
            builder.getKindMap().getCharacterBitsize(charTy.getFKind()) / 8;
        aiir::Value charSz =
            builder.createIntegerConstant(loc, idxTy, kindBytes);
        aiir::Value len =
            aiir::arith::DivSIOp::create(builder, loc, eleSz, charSz);
        return {len};
      }
    }
  }
  return {};
}

// fir::getTypeParams() will get the type parameters from the extended value.
// When the extended value is a BoxValue or MutableBoxValue, it may be necessary
// to generate code, so this factory function handles those cases.
// TODO: fix the inverted type tests, etc.
llvm::SmallVector<aiir::Value>
fir::factory::getTypeParams(aiir::Location loc, fir::FirOpBuilder &builder,
                            const fir::ExtendedValue &exv) {
  auto handleBoxed = [&](const auto &box) -> llvm::SmallVector<aiir::Value> {
    if (box.isCharacter())
      return {fir::factory::readCharLen(builder, loc, exv)};
    if (box.isDerivedWithLenParameters()) {
      // This should generate code to read the type parameters from the box.
      // This requires some consideration however as MutableBoxValues need to be
      // in a sane state to be provide the correct values.
      TODO(loc, "derived type with type parameters");
    }
    return {};
  };
  // Intentionally reuse the original code path to get type parameters for the
  // cases that were supported rather than introduce a new path.
  return exv.match(
      [&](const fir::BoxValue &box) { return handleBoxed(box); },
      [&](const fir::MutableBoxValue &box) { return handleBoxed(box); },
      [&](const auto &) { return fir::getTypeParams(exv); });
}

llvm::SmallVector<aiir::Value>
fir::factory::getTypeParams(aiir::Location loc, fir::FirOpBuilder &builder,
                            fir::ArrayLoadOp load) {
  aiir::Type memTy = load.getMemref().getType();
  if (auto boxTy = aiir::dyn_cast<fir::BaseBoxType>(memTy))
    return getFromBox(loc, builder, boxTy, load.getMemref());
  return load.getTypeparams();
}

std::string fir::factory::uniqueCGIdent(llvm::StringRef prefix,
                                        llvm::StringRef name) {
  // For "long" identifiers use a hash value
  if (name.size() > nameLengthHashSize) {
    llvm::MD5 hash;
    hash.update(name);
    llvm::MD5::MD5Result result;
    hash.final(result);
    llvm::SmallString<32> str;
    llvm::MD5::stringifyResult(result, str);
    std::string hashName = prefix.str();
    hashName.append("X").append(str.c_str());
    return fir::NameUniquer::doGenerated(hashName);
  }
  // "Short" identifiers use a reversible hex string
  std::string nm = prefix.str();
  return fir::NameUniquer::doGenerated(
      nm.append("X").append(llvm::toHex(name)));
}

aiir::Value fir::factory::locationToFilename(fir::FirOpBuilder &builder,
                                             aiir::Location loc) {
  if (auto flc = aiir::dyn_cast<aiir::FileLineColLoc>(loc)) {
    // must be encoded as asciiz, C string
    auto fn = flc.getFilename().str() + '\0';
    return fir::getBase(createStringLiteral(builder, loc, fn));
  }
  return builder.createNullConstant(loc);
}

aiir::Value fir::factory::locationToLineNo(fir::FirOpBuilder &builder,
                                           aiir::Location loc,
                                           aiir::Type type) {
  if (auto flc = aiir::dyn_cast<aiir::FileLineColLoc>(loc))
    return builder.createIntegerConstant(loc, type, flc.getLine());
  return builder.createIntegerConstant(loc, type, 0);
}

fir::ExtendedValue fir::factory::createStringLiteral(fir::FirOpBuilder &builder,
                                                     aiir::Location loc,
                                                     llvm::StringRef str) {
  std::string globalName = fir::factory::uniqueCGIdent("cl", str);
  auto type = fir::CharacterType::get(builder.getContext(), 1, str.size());
  auto global = builder.getNamedGlobal(globalName);
  if (!global)
    global = builder.createGlobalConstant(
        loc, type, globalName,
        [&](fir::FirOpBuilder &builder) {
          auto stringLitOp = builder.createStringLitOp(loc, str);
          fir::HasValueOp::create(builder, loc, stringLitOp);
        },
        builder.createLinkOnceLinkage());
  auto addr = fir::AddrOfOp::create(builder, loc, global.resultType(),
                                    global.getSymbol());
  auto len = builder.createIntegerConstant(
      loc, builder.getCharacterLengthType(), str.size());
  return fir::CharBoxValue{addr, len};
}

llvm::SmallVector<aiir::Value>
fir::factory::createExtents(fir::FirOpBuilder &builder, aiir::Location loc,
                            fir::SequenceType seqTy) {
  llvm::SmallVector<aiir::Value> extents;
  auto idxTy = builder.getIndexType();
  for (auto ext : seqTy.getShape())
    extents.emplace_back(
        ext == fir::SequenceType::getUnknownExtent()
            ? fir::UndefOp::create(builder, loc, idxTy).getResult()
            : builder.createIntegerConstant(loc, idxTy, ext));
  return extents;
}

// FIXME: This needs some work. To correctly determine the extended value of a
// component, one needs the base object, its type, and its type parameters. (An
// alternative would be to provide an already computed address of the final
// component rather than the base object's address, the point being the result
// will require the address of the final component to create the extended
// value.) One further needs the full path of components being applied. One
// needs to apply type-based expressions to type parameters along this said
// path. (See applyPathToType for a type-only derivation.) Finally, one needs to
// compose the extended value of the terminal component, including all of its
// parameters: array lower bounds expressions, extents, type parameters, etc.
// Any of these properties may be deferred until runtime in Fortran. This
// operation may therefore generate a sizeable block of IR, including calls to
// type-based helper functions, so caching the result of this operation in the
// client would be advised as well.
fir::ExtendedValue fir::factory::componentToExtendedValue(
    fir::FirOpBuilder &builder, aiir::Location loc, aiir::Value component) {
  auto fieldTy = component.getType();
  if (auto ty = fir::dyn_cast_ptrEleTy(fieldTy))
    fieldTy = ty;
  if (aiir::isa<fir::BaseBoxType>(fieldTy)) {
    llvm::SmallVector<aiir::Value> nonDeferredTypeParams;
    auto eleTy = fir::unwrapSequenceType(fir::dyn_cast_ptrOrBoxEleTy(fieldTy));
    if (auto charTy = aiir::dyn_cast<fir::CharacterType>(eleTy)) {
      auto lenTy = builder.getCharacterLengthType();
      if (charTy.hasConstantLen())
        nonDeferredTypeParams.emplace_back(
            builder.createIntegerConstant(loc, lenTy, charTy.getLen()));
      // TODO: Starting, F2003, the dynamic character length might be dependent
      // on a PDT length parameter. There is no way to make a difference with
      // deferred length here yet.
    }
    if (auto recTy = aiir::dyn_cast<fir::RecordType>(eleTy))
      if (recTy.getNumLenParams() > 0)
        TODO(loc, "allocatable and pointer components non deferred length "
                  "parameters");

    return fir::MutableBoxValue(component, nonDeferredTypeParams,
                                /*mutableProperties=*/{});
  }
  llvm::SmallVector<aiir::Value> extents;
  if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(fieldTy)) {
    fieldTy = seqTy.getEleTy();
    auto idxTy = builder.getIndexType();
    for (auto extent : seqTy.getShape()) {
      if (extent == fir::SequenceType::getUnknownExtent())
        TODO(loc, "array component shape depending on length parameters");
      extents.emplace_back(builder.createIntegerConstant(loc, idxTy, extent));
    }
  }
  if (auto charTy = aiir::dyn_cast<fir::CharacterType>(fieldTy)) {
    auto cstLen = charTy.getLen();
    if (cstLen == fir::CharacterType::unknownLen())
      TODO(loc, "get character component length from length type parameters");
    auto len = builder.createIntegerConstant(
        loc, builder.getCharacterLengthType(), cstLen);
    if (!extents.empty())
      return fir::CharArrayBoxValue{component, len, extents};
    return fir::CharBoxValue{component, len};
  }
  if (auto recordTy = aiir::dyn_cast<fir::RecordType>(fieldTy))
    if (recordTy.getNumLenParams() != 0)
      TODO(loc,
           "lower component ref that is a derived type with length parameter");
  if (!extents.empty())
    return fir::ArrayBoxValue{component, extents};
  return component;
}

fir::ExtendedValue fir::factory::arrayElementToExtendedValue(
    fir::FirOpBuilder &builder, aiir::Location loc,
    const fir::ExtendedValue &array, aiir::Value element) {
  return array.match(
      [&](const fir::CharBoxValue &cb) -> fir::ExtendedValue {
        return cb.clone(element);
      },
      [&](const fir::CharArrayBoxValue &bv) -> fir::ExtendedValue {
        return bv.cloneElement(element);
      },
      [&](const fir::BoxValue &box) -> fir::ExtendedValue {
        if (box.isCharacter()) {
          auto len = fir::factory::readCharLen(builder, loc, box);
          return fir::CharBoxValue{element, len};
        }
        if (box.isDerivedWithLenParameters())
          TODO(loc, "get length parameters from derived type BoxValue");
        if (box.isPolymorphic()) {
          return fir::PolymorphicValue(element, fir::getBase(box));
        }
        return element;
      },
      [&](const fir::ArrayBoxValue &box) -> fir::ExtendedValue {
        if (box.getSourceBox())
          return fir::PolymorphicValue(element, box.getSourceBox());
        return element;
      },
      [&](const auto &) -> fir::ExtendedValue { return element; });
}

fir::ExtendedValue fir::factory::arraySectionElementToExtendedValue(
    fir::FirOpBuilder &builder, aiir::Location loc,
    const fir::ExtendedValue &array, aiir::Value element, aiir::Value slice) {
  if (!slice)
    return arrayElementToExtendedValue(builder, loc, array, element);
  auto sliceOp = aiir::dyn_cast_or_null<fir::SliceOp>(slice.getDefiningOp());
  assert(sliceOp && "slice must be a sliceOp");
  if (sliceOp.getFields().empty())
    return arrayElementToExtendedValue(builder, loc, array, element);
  // For F95, using componentToExtendedValue will work, but when PDTs are
  // lowered. It will be required to go down the slice to propagate the length
  // parameters.
  return fir::factory::componentToExtendedValue(builder, loc, element);
}

void fir::factory::genScalarAssignment(
    fir::FirOpBuilder &builder, aiir::Location loc,
    const fir::ExtendedValue &lhs, const fir::ExtendedValue &rhs,
    bool needFinalization, bool isTemporaryLHS, aiir::ArrayAttr accessGroups) {
  assert(lhs.rank() == 0 && rhs.rank() == 0 && "must be scalars");
  auto type = fir::unwrapSequenceType(
      fir::unwrapPassByRefType(fir::getBase(lhs).getType()));
  if (aiir::isa<fir::CharacterType>(type)) {
    const fir::CharBoxValue *toChar = lhs.getCharBox();
    const fir::CharBoxValue *fromChar = rhs.getCharBox();
    assert(toChar && fromChar);
    fir::factory::CharacterExprHelper helper{builder, loc};
    helper.createAssign(fir::ExtendedValue{*toChar},
                        fir::ExtendedValue{*fromChar});
  } else if (aiir::isa<fir::RecordType>(type)) {
    fir::factory::genRecordAssignment(builder, loc, lhs, rhs, needFinalization,
                                      isTemporaryLHS);
  } else {
    assert(!fir::hasDynamicSize(type));
    auto rhsVal = fir::getBase(rhs);
    if (fir::isa_ref_type(rhsVal.getType()))
      rhsVal = fir::LoadOp::create(builder, loc, rhsVal);
    aiir::Value lhsAddr = fir::getBase(lhs);
    rhsVal = builder.createConvert(loc, fir::unwrapRefType(lhsAddr.getType()),
                                   rhsVal);
    fir::StoreOp store = fir::StoreOp::create(builder, loc, rhsVal, lhsAddr);
    if (accessGroups)
      store.setAccessGroupsAttr(accessGroups);
  }
}

static void genComponentByComponentAssignment(fir::FirOpBuilder &builder,
                                              aiir::Location loc,
                                              const fir::ExtendedValue &lhs,
                                              const fir::ExtendedValue &rhs,
                                              bool isTemporaryLHS) {
  auto lbaseType = fir::unwrapPassByRefType(fir::getBase(lhs).getType());
  auto lhsType = aiir::dyn_cast<fir::RecordType>(lbaseType);
  assert(lhsType && "lhs must be a scalar record type");
  auto rbaseType = fir::unwrapPassByRefType(fir::getBase(rhs).getType());
  auto rhsType = aiir::dyn_cast<fir::RecordType>(rbaseType);
  assert(rhsType && "rhs must be a scalar record type");
  auto fieldIndexType = fir::FieldType::get(lhsType.getContext());
  for (auto [lhsPair, rhsPair] :
       llvm::zip(lhsType.getTypeList(), rhsType.getTypeList())) {
    auto &[lFieldName, lFieldTy] = lhsPair;
    auto &[rFieldName, rFieldTy] = rhsPair;
    assert(!fir::hasDynamicSize(lFieldTy) && !fir::hasDynamicSize(rFieldTy));
    aiir::Value rField =
        fir::FieldIndexOp::create(builder, loc, fieldIndexType, rFieldName,
                                  rhsType, fir::getTypeParams(rhs));
    auto rFieldRefType = builder.getRefType(rFieldTy);
    aiir::Value fromCoor = fir::CoordinateOp::create(
        builder, loc, rFieldRefType, fir::getBase(rhs), rField);
    aiir::Value field =
        fir::FieldIndexOp::create(builder, loc, fieldIndexType, lFieldName,
                                  lhsType, fir::getTypeParams(lhs));
    auto fieldRefType = builder.getRefType(lFieldTy);
    aiir::Value toCoor = fir::CoordinateOp::create(builder, loc, fieldRefType,
                                                   fir::getBase(lhs), field);
    std::optional<fir::DoLoopOp> outerLoop;
    if (auto sequenceType = aiir::dyn_cast<fir::SequenceType>(lFieldTy)) {
      // Create loops to assign array components elements by elements.
      // Note that, since these are components, they either do not overlap,
      // or are the same and exactly overlap. They also have compile time
      // constant shapes.
      aiir::Type idxTy = builder.getIndexType();
      llvm::SmallVector<aiir::Value> indices;
      aiir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
      aiir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
      for (auto extent : llvm::reverse(sequenceType.getShape())) {
        // TODO: add zero size test !
        aiir::Value ub = builder.createIntegerConstant(loc, idxTy, extent - 1);
        auto loop = fir::DoLoopOp::create(builder, loc, zero, ub, one);
        if (!outerLoop)
          outerLoop = loop;
        indices.push_back(loop.getInductionVar());
        builder.setInsertionPointToStart(loop.getBody());
      }
      // Set indices in column-major order.
      std::reverse(indices.begin(), indices.end());
      auto elementRefType = builder.getRefType(sequenceType.getEleTy());
      toCoor = fir::CoordinateOp::create(builder, loc, elementRefType, toCoor,
                                         indices);
      fromCoor = fir::CoordinateOp::create(builder, loc, elementRefType,
                                           fromCoor, indices);
    }
    if (auto fieldEleTy = fir::unwrapSequenceType(lFieldTy);
        aiir::isa<fir::BaseBoxType>(fieldEleTy)) {
      assert(aiir::isa<fir::PointerType>(
                 aiir::cast<fir::BaseBoxType>(fieldEleTy).getEleTy()) &&
             "allocatable members require deep copy");
      auto fromPointerValue = fir::LoadOp::create(builder, loc, fromCoor);
      auto castTo = builder.createConvert(loc, fieldEleTy, fromPointerValue);
      fir::StoreOp::create(builder, loc, castTo, toCoor);
    } else {
      auto from =
          fir::factory::componentToExtendedValue(builder, loc, fromCoor);
      auto to = fir::factory::componentToExtendedValue(builder, loc, toCoor);
      // If LHS finalization is needed it is expected to be done
      // for the parent record, so that component-by-component
      // assignments may avoid finalization calls.
      fir::factory::genScalarAssignment(builder, loc, to, from,
                                        /*needFinalization=*/false,
                                        isTemporaryLHS);
    }
    if (outerLoop)
      builder.setInsertionPointAfter(*outerLoop);
  }
}

/// Can the assignment of this record type be implement with a simple memory
/// copy (it requires no deep copy or user defined assignment of components )?
static bool recordTypeCanBeMemCopied(fir::RecordType recordType) {
  // c_devptr type is a special case. It has a nested c_ptr field but we know it
  // can be copied directly.
  if (fir::isa_builtin_c_devptr_type(recordType))
    return true;
  if (fir::hasDynamicSize(recordType))
    return false;
  for (auto [_, fieldType] : recordType.getTypeList()) {
    // Derived type component may have user assignment (so far, we cannot tell
    // in FIR, so assume it is always the case, TODO: get the actual info).
    if (aiir::isa<fir::RecordType>(fir::unwrapSequenceType(fieldType)) &&
        !fir::isa_builtin_c_devptr_type(fir::unwrapSequenceType(fieldType)))
      return false;
    // Allocatable components need deep copy.
    if (auto boxType = aiir::dyn_cast<fir::BaseBoxType>(fieldType))
      if (aiir::isa<fir::HeapType>(boxType.getEleTy()))
        return false;
  }
  // Constant size components without user defined assignment and pointers can
  // be memcopied.
  return true;
}

static bool mayHaveFinalizer(fir::RecordType recordType,
                             fir::FirOpBuilder &builder) {
  if (auto typeInfo = builder.getModule().lookupSymbol<fir::TypeInfoOp>(
          recordType.getName()))
    return !typeInfo.getNoFinal();
  // No info, be pessimistic.
  return true;
}

void fir::factory::genRecordAssignment(fir::FirOpBuilder &builder,
                                       aiir::Location loc,
                                       const fir::ExtendedValue &lhs,
                                       const fir::ExtendedValue &rhs,
                                       bool needFinalization,
                                       bool isTemporaryLHS) {
  assert(lhs.rank() == 0 && rhs.rank() == 0 && "assume scalar assignment");
  auto baseTy = fir::dyn_cast_ptrOrBoxEleTy(fir::getBase(lhs).getType());
  assert(baseTy && "must be a memory type");
  // Box operands may be polymorphic, it is not entirely clear from 10.2.1.3
  // if the assignment is performed on the dynamic of declared type. Use the
  // runtime assuming it is performed on the dynamic type.
  bool hasBoxOperands =
      aiir::isa<fir::BaseBoxType>(fir::getBase(lhs).getType()) ||
      aiir::isa<fir::BaseBoxType>(fir::getBase(rhs).getType());
  auto recTy = aiir::dyn_cast<fir::RecordType>(baseTy);
  assert(recTy && "must be a record type");

  // Use alias analysis to guard the fast path.
  fir::AliasAnalysis aa;
  // Aliased SEQUENCE types must take the conservative (slow) path.
  bool disjoint = isTemporaryLHS || !recTy.isSequence() ||
                  (aa.alias(fir::getBase(lhs), fir::getBase(rhs)) ==
                   aiir::AliasResult::NoAlias);
  if ((needFinalization && mayHaveFinalizer(recTy, builder)) ||
      hasBoxOperands || !recordTypeCanBeMemCopied(recTy) || !disjoint) {
    auto to = fir::getBase(builder.createBox(loc, lhs));
    auto from = fir::getBase(builder.createBox(loc, rhs));
    // The runtime entry point may modify the LHS descriptor if it is
    // an allocatable. Allocatable assignment is handle elsewhere in lowering,
    // so just create a fir.ref<fir.box<>> from the fir.box to comply with the
    // runtime interface, but assume the fir.box is unchanged.
    // TODO: does this holds true with polymorphic entities ?
    auto toMutableBox = builder.createTemporary(loc, to.getType());
    fir::StoreOp::create(builder, loc, to, toMutableBox);
    if (isTemporaryLHS)
      fir::runtime::genAssignTemporary(builder, loc, toMutableBox, from);
    else
      fir::runtime::genAssign(builder, loc, toMutableBox, from);
    return;
  }

  // Otherwise, the derived type has compile time constant size and for which
  // the component by component assignment can be replaced by a memory copy.
  // Since we do not know the size of the derived type in lowering, do a
  // component by component assignment. Note that a single fir.load/fir.store
  // could be used on "small" record types, but as the type size grows, this
  // leads to issues in LLVM (long compile times, long IR files, and even
  // asserts at some point). Since there is no good size boundary, just always
  // use component by component assignment here.
  genComponentByComponentAssignment(builder, loc, lhs, rhs, isTemporaryLHS);
}

aiir::TupleType
fir::factory::getRaggedArrayHeaderType(fir::FirOpBuilder &builder) {
  aiir::IntegerType i64Ty = builder.getIntegerType(64);
  auto arrTy = fir::SequenceType::get(builder.getIntegerType(8), 1);
  auto buffTy = fir::HeapType::get(arrTy);
  auto extTy = fir::SequenceType::get(i64Ty, 1);
  auto shTy = fir::HeapType::get(extTy);
  return aiir::TupleType::get(builder.getContext(), {i64Ty, buffTy, shTy});
}

aiir::Value fir::factory::genLenOfCharacter(
    fir::FirOpBuilder &builder, aiir::Location loc, fir::ArrayLoadOp arrLoad,
    llvm::ArrayRef<aiir::Value> path, llvm::ArrayRef<aiir::Value> substring) {
  llvm::SmallVector<aiir::Value> typeParams(arrLoad.getTypeparams());
  return genLenOfCharacter(builder, loc,
                           aiir::cast<fir::SequenceType>(arrLoad.getType()),
                           arrLoad.getMemref(), typeParams, path, substring);
}

aiir::Value fir::factory::genLenOfCharacter(
    fir::FirOpBuilder &builder, aiir::Location loc, fir::SequenceType seqTy,
    aiir::Value memref, llvm::ArrayRef<aiir::Value> typeParams,
    llvm::ArrayRef<aiir::Value> path, llvm::ArrayRef<aiir::Value> substring) {
  auto idxTy = builder.getIndexType();
  auto zero = builder.createIntegerConstant(loc, idxTy, 0);
  auto saturatedDiff = [&](aiir::Value lower, aiir::Value upper) {
    auto diff = aiir::arith::SubIOp::create(builder, loc, upper, lower);
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    auto size = aiir::arith::AddIOp::create(builder, loc, diff, one);
    auto cmp = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::sgt, size, zero);
    return aiir::arith::SelectOp::create(builder, loc, cmp, size, zero);
  };
  if (substring.size() == 2) {
    auto upper = builder.createConvert(loc, idxTy, substring.back());
    auto lower = builder.createConvert(loc, idxTy, substring.front());
    return saturatedDiff(lower, upper);
  }
  auto lower = zero;
  if (substring.size() == 1)
    lower = builder.createConvert(loc, idxTy, substring.front());
  auto eleTy = fir::applyPathToType(seqTy, path);
  if (!fir::hasDynamicSize(eleTy)) {
    if (auto charTy = aiir::dyn_cast<fir::CharacterType>(eleTy)) {
      // Use LEN from the type.
      return builder.createIntegerConstant(loc, idxTy, charTy.getLen());
    }
    // Do we need to support !fir.array<!fir.char<k,n>>?
    fir::emitFatalError(loc,
                        "application of path did not result in a !fir.char");
  }
  if (fir::isa_box_type(memref.getType())) {
    if (aiir::isa<fir::BoxCharType>(memref.getType()))
      return fir::BoxCharLenOp::create(builder, loc, idxTy, memref);
    if (aiir::isa<fir::BoxType>(memref.getType()))
      return CharacterExprHelper(builder, loc).readLengthFromBox(memref);
    fir::emitFatalError(loc, "memref has wrong type");
  }
  if (typeParams.empty()) {
    fir::emitFatalError(loc, "array_load must have typeparams");
  }
  if (fir::isa_char(seqTy.getEleTy())) {
    assert(typeParams.size() == 1 && "too many typeparams");
    return typeParams.front();
  }
  TODO(loc, "LEN of character must be computed at runtime");
}

aiir::Value fir::factory::createZeroValue(fir::FirOpBuilder &builder,
                                          aiir::Location loc, aiir::Type type) {
  aiir::Type i1 = builder.getIntegerType(1);
  if (aiir::isa<fir::LogicalType>(type) || type == i1)
    return builder.createConvert(loc, type, builder.createBool(loc, false));
  if (fir::isa_integer(type))
    return builder.createIntegerConstant(loc, type, 0);
  if (fir::isa_real(type))
    return builder.createRealZeroConstant(loc, type);
  if (fir::isa_complex(type)) {
    fir::factory::Complex complexHelper(builder, loc);
    aiir::Type partType = complexHelper.getComplexPartType(type);
    aiir::Value zeroPart = builder.createRealZeroConstant(loc, partType);
    return complexHelper.createComplex(type, zeroPart, zeroPart);
  }
  fir::emitFatalError(loc, "internal: trying to generate zero value of non "
                           "numeric or logical type");
}

aiir::Value fir::factory::createOneValue(fir::FirOpBuilder &builder,
                                         aiir::Location loc, aiir::Type type) {
  aiir::Type i1 = builder.getIntegerType(1);
  if (aiir::isa<fir::LogicalType>(type) || type == i1)
    return builder.createConvert(loc, type, builder.createBool(loc, true));
  if (fir::isa_integer(type))
    return builder.createIntegerConstant(loc, type, 1);
  if (fir::isa_real(type))
    return builder.createRealOneConstant(loc, type);
  if (fir::isa_complex(type)) {
    fir::factory::Complex complexHelper(builder, loc);
    aiir::Type partType = complexHelper.getComplexPartType(type);
    aiir::Value realPart = builder.createRealOneConstant(loc, partType);
    aiir::Value imagPart = builder.createRealZeroConstant(loc, partType);
    return complexHelper.createComplex(type, realPart, imagPart);
  }
  fir::emitFatalError(loc, "internal: trying to generate one value of non "
                           "numeric or logical type");
}

std::optional<std::int64_t>
fir::factory::getExtentFromTriplet(aiir::Value lb, aiir::Value ub,
                                   aiir::Value stride) {
  std::function<std::optional<std::int64_t>(aiir::Value)> getConstantValue =
      [&](aiir::Value value) -> std::optional<std::int64_t> {
    if (auto valInt = fir::getIntIfConstant(value))
      return *valInt;
    auto *definingOp = value.getDefiningOp();
    if (aiir::isa_and_nonnull<fir::ConvertOp>(definingOp)) {
      auto valOp = aiir::dyn_cast<fir::ConvertOp>(definingOp);
      return getConstantValue(valOp.getValue());
    }
    return {};
  };
  if (auto lbInt = getConstantValue(lb)) {
    if (auto ubInt = getConstantValue(ub)) {
      if (auto strideInt = getConstantValue(stride)) {
        if (*strideInt != 0) {
          std::int64_t extent = 1 + (*ubInt - *lbInt) / *strideInt;
          if (extent > 0)
            return extent;
        }
      }
    }
  }
  return {};
}

aiir::Value fir::factory::genMaxWithZero(fir::FirOpBuilder &builder,
                                         aiir::Location loc, aiir::Value value,
                                         aiir::Value zero) {
  if (aiir::Operation *definingOp = value.getDefiningOp())
    if (auto cst = aiir::dyn_cast<aiir::arith::ConstantOp>(definingOp))
      if (auto intAttr = aiir::dyn_cast<aiir::IntegerAttr>(cst.getValue()))
        return intAttr.getInt() > 0 ? value : zero;
  aiir::Value valueIsGreater = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::sgt, value, zero);
  return aiir::arith::SelectOp::create(builder, loc, valueIsGreater, value,
                                       zero);
}

aiir::Value fir::factory::genMaxWithZero(fir::FirOpBuilder &builder,
                                         aiir::Location loc,
                                         aiir::Value value) {
  aiir::Value zero = builder.createIntegerConstant(loc, value.getType(), 0);
  return genMaxWithZero(builder, loc, value, zero);
}

aiir::Value fir::factory::computeExtent(fir::FirOpBuilder &builder,
                                        aiir::Location loc, aiir::Value lb,
                                        aiir::Value ub, aiir::Value zero,
                                        aiir::Value one) {
  aiir::Type type = lb.getType();
  // Let the folder deal with the common `ub - <const> + 1` case.
  auto diff = aiir::arith::SubIOp::create(builder, loc, type, ub, lb);
  auto rawExtent = aiir::arith::AddIOp::create(builder, loc, type, diff, one);
  return fir::factory::genMaxWithZero(builder, loc, rawExtent, zero);
}
aiir::Value fir::factory::computeExtent(fir::FirOpBuilder &builder,
                                        aiir::Location loc, aiir::Value lb,
                                        aiir::Value ub) {
  aiir::Type type = lb.getType();
  aiir::Value one = builder.createIntegerConstant(loc, type, 1);
  aiir::Value zero = builder.createIntegerConstant(loc, type, 0);
  return computeExtent(builder, loc, lb, ub, zero, one);
}

static std::pair<aiir::Value, aiir::Type>
genCPtrOrCFunptrFieldIndex(fir::FirOpBuilder &builder, aiir::Location loc,
                           aiir::Type cptrTy) {
  auto recTy = aiir::cast<fir::RecordType>(cptrTy);
  assert(recTy.getTypeList().size() == 1);
  auto addrFieldName = recTy.getTypeList()[0].first;
  aiir::Type addrFieldTy = recTy.getTypeList()[0].second;
  auto fieldIndexType = fir::FieldType::get(cptrTy.getContext());
  aiir::Value addrFieldIndex = fir::FieldIndexOp::create(
      builder, loc, fieldIndexType, addrFieldName, recTy,
      /*typeParams=*/aiir::ValueRange{});
  return {addrFieldIndex, addrFieldTy};
}

aiir::Value fir::factory::genCPtrOrCFunptrAddr(fir::FirOpBuilder &builder,
                                               aiir::Location loc,
                                               aiir::Value cPtr,
                                               aiir::Type ty) {
  auto [addrFieldIndex, addrFieldTy] =
      genCPtrOrCFunptrFieldIndex(builder, loc, ty);
  return fir::CoordinateOp::create(
      builder, loc, builder.getRefType(addrFieldTy), cPtr, addrFieldIndex);
}

aiir::Value fir::factory::genCDevPtrAddr(fir::FirOpBuilder &builder,
                                         aiir::Location loc,
                                         aiir::Value cDevPtr, aiir::Type ty) {
  auto recTy = aiir::cast<fir::RecordType>(ty);
  assert(recTy.getTypeList().size() == 1);
  auto cptrFieldName = recTy.getTypeList()[0].first;
  aiir::Type cptrFieldTy = recTy.getTypeList()[0].second;
  auto fieldIndexType = fir::FieldType::get(ty.getContext());
  aiir::Value cptrFieldIndex = fir::FieldIndexOp::create(
      builder, loc, fieldIndexType, cptrFieldName, recTy,
      /*typeParams=*/aiir::ValueRange{});
  auto cptrCoord = fir::CoordinateOp::create(
      builder, loc, builder.getRefType(cptrFieldTy), cDevPtr, cptrFieldIndex);
  auto [addrFieldIndex, addrFieldTy] =
      genCPtrOrCFunptrFieldIndex(builder, loc, cptrFieldTy);
  return fir::CoordinateOp::create(
      builder, loc, builder.getRefType(addrFieldTy), cptrCoord, addrFieldIndex);
}

aiir::Value fir::factory::genCPtrOrCFunptrValue(fir::FirOpBuilder &builder,
                                                aiir::Location loc,
                                                aiir::Value cPtr) {
  aiir::Type cPtrTy = fir::unwrapRefType(cPtr.getType());
  if (fir::isa_builtin_cdevptr_type(cPtrTy)) {
    // Unwrap c_ptr from c_devptr.
    auto [addrFieldIndex, addrFieldTy] =
        genCPtrOrCFunptrFieldIndex(builder, loc, cPtrTy);
    aiir::Value cPtrCoor;
    if (fir::isa_ref_type(cPtr.getType())) {
      cPtrCoor = fir::CoordinateOp::create(
          builder, loc, builder.getRefType(addrFieldTy), cPtr, addrFieldIndex);
    } else {
      auto arrayAttr = builder.getArrayAttr(
          {builder.getIntegerAttr(builder.getIndexType(), 0)});
      cPtrCoor = fir::ExtractValueOp::create(builder, loc, addrFieldTy, cPtr,
                                             arrayAttr);
    }
    return genCPtrOrCFunptrValue(builder, loc, cPtrCoor);
  }

  if (fir::isa_ref_type(cPtr.getType())) {
    aiir::Value cPtrAddr =
        fir::factory::genCPtrOrCFunptrAddr(builder, loc, cPtr, cPtrTy);
    return fir::LoadOp::create(builder, loc, cPtrAddr);
  }
  auto [addrFieldIndex, addrFieldTy] =
      genCPtrOrCFunptrFieldIndex(builder, loc, cPtrTy);
  auto arrayAttr =
      builder.getArrayAttr({builder.getIntegerAttr(builder.getIndexType(), 0)});
  return fir::ExtractValueOp::create(builder, loc, addrFieldTy, cPtr,
                                     arrayAttr);
}

fir::BoxValue fir::factory::createBoxValue(fir::FirOpBuilder &builder,
                                           aiir::Location loc,
                                           const fir::ExtendedValue &exv) {
  if (auto *boxValue = exv.getBoxOf<fir::BoxValue>())
    return *boxValue;
  aiir::Value box = builder.createBox(loc, exv);
  llvm::SmallVector<aiir::Value> lbounds;
  llvm::SmallVector<aiir::Value> explicitTypeParams;
  exv.match(
      [&](const fir::ArrayBoxValue &box) {
        lbounds.append(box.getLBounds().begin(), box.getLBounds().end());
      },
      [&](const fir::CharArrayBoxValue &box) {
        lbounds.append(box.getLBounds().begin(), box.getLBounds().end());
        explicitTypeParams.emplace_back(box.getLen());
      },
      [&](const fir::CharBoxValue &box) {
        explicitTypeParams.emplace_back(box.getLen());
      },
      [&](const fir::MutableBoxValue &x) {
        if (x.rank() > 0) {
          // The resulting box lbounds must be coming from the mutable box.
          fir::ExtendedValue boxVal =
              fir::factory::genMutableBoxRead(builder, loc, x);
          // Make sure we do not recurse infinitely.
          if (boxVal.getBoxOf<fir::MutableBoxValue>())
            fir::emitFatalError(loc, "mutable box read cannot be mutable box");
          fir::BoxValue box =
              fir::factory::createBoxValue(builder, loc, boxVal);
          lbounds.append(box.getLBounds().begin(), box.getLBounds().end());
        }
        explicitTypeParams.append(x.nonDeferredLenParams().begin(),
                                  x.nonDeferredLenParams().end());
      },
      [](const auto &) {});
  return fir::BoxValue(box, lbounds, explicitTypeParams);
}

aiir::Value fir::factory::createNullBoxProc(fir::FirOpBuilder &builder,
                                            aiir::Location loc,
                                            aiir::Type boxType) {
  auto boxTy{aiir::dyn_cast<fir::BoxProcType>(boxType)};
  if (!boxTy)
    fir::emitFatalError(loc, "Procedure pointer must be of BoxProcType");
  auto boxEleTy{fir::unwrapRefType(boxTy.getEleTy())};
  aiir::Value initVal{fir::ZeroOp::create(builder, loc, boxEleTy)};
  return fir::EmboxProcOp::create(builder, loc, boxTy, initVal);
}

void fir::factory::setInternalLinkage(aiir::func::FuncOp func) {
  auto internalLinkage = aiir::LLVM::linkage::Linkage::Internal;
  auto linkage =
      aiir::LLVM::LinkageAttr::get(func->getContext(), internalLinkage);
  func->setAttr("llvm.linkage", linkage);
}

uint64_t
fir::factory::getAllocaAddressSpace(const aiir::DataLayout *dataLayout) {
  if (dataLayout)
    if (aiir::Attribute addrSpace = dataLayout->getAllocaMemorySpace())
      return aiir::cast<aiir::IntegerAttr>(addrSpace).getUInt();
  return 0;
}

llvm::SmallVector<aiir::Value>
fir::factory::deduceOptimalExtents(aiir::ValueRange extents1,
                                   aiir::ValueRange extents2) {
  llvm::SmallVector<aiir::Value> extents;
  extents.reserve(extents1.size());
  for (auto [extent1, extent2] : llvm::zip(extents1, extents2)) {
    if (!fir::getIntIfConstant(extent1) && fir::getIntIfConstant(extent2))
      extents.push_back(extent2);
    else
      extents.push_back(extent1);
  }
  return extents;
}

uint64_t fir::factory::getGlobalAddressSpace(aiir::DataLayout *dataLayout) {
  if (dataLayout)
    if (aiir::Attribute addrSpace = dataLayout->getGlobalMemorySpace())
      return aiir::cast<aiir::IntegerAttr>(addrSpace).getUInt();
  return 0;
}

uint64_t fir::factory::getProgramAddressSpace(aiir::DataLayout *dataLayout) {
  if (dataLayout)
    if (aiir::Attribute addrSpace = dataLayout->getProgramMemorySpace())
      return aiir::cast<aiir::IntegerAttr>(addrSpace).getUInt();
  return 0;
}

llvm::SmallVector<aiir::Value> fir::factory::updateRuntimeExtentsForEmptyArrays(
    fir::FirOpBuilder &builder, aiir::Location loc, aiir::ValueRange extents) {
  if (extents.size() <= 1)
    return extents;

  aiir::Type i1Type = builder.getI1Type();
  aiir::Value isEmpty = createZeroValue(builder, loc, i1Type);

  llvm::SmallVector<aiir::Value, Fortran::common::maxRank> zeroes;
  for (aiir::Value extent : extents) {
    aiir::Type type = extent.getType();
    aiir::Value zero = createZeroValue(builder, loc, type);
    zeroes.push_back(zero);
    aiir::Value isZero = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::eq, extent, zero);
    isEmpty = aiir::arith::OrIOp::create(builder, loc, isEmpty, isZero);
  }

  llvm::SmallVector<aiir::Value> newExtents;
  for (auto [zero, extent] : llvm::zip_equal(zeroes, extents)) {
    newExtents.push_back(
        aiir::arith::SelectOp::create(builder, loc, isEmpty, zero, extent));
  }
  return newExtents;
}

void fir::factory::genDimInfoFromBox(
    fir::FirOpBuilder &builder, aiir::Location loc, aiir::Value box,
    llvm::SmallVectorImpl<aiir::Value> *lbounds,
    llvm::SmallVectorImpl<aiir::Value> *extents,
    llvm::SmallVectorImpl<aiir::Value> *strides) {
  auto boxType = aiir::dyn_cast<fir::BaseBoxType>(box.getType());
  assert(boxType && "must be a box");
  if (!lbounds && !extents && !strides)
    return;

  unsigned rank = fir::getBoxRank(boxType);
  assert(!boxType.isAssumedRank() && "must be an array of known rank");
  aiir::Type idxTy = builder.getIndexType();
  for (unsigned i = 0; i < rank; ++i) {
    aiir::Value dim = builder.createIntegerConstant(loc, idxTy, i);
    auto dimInfo =
        fir::BoxDimsOp::create(builder, loc, idxTy, idxTy, idxTy, box, dim);
    if (lbounds)
      lbounds->push_back(dimInfo.getLowerBound());
    if (extents)
      extents->push_back(dimInfo.getExtent());
    if (strides)
      strides->push_back(dimInfo.getByteStride());
  }
}

aiir::Value fir::factory::genLifetimeStart(aiir::OpBuilder &builder,
                                           aiir::Location loc,
                                           fir::AllocaOp alloc,
                                           const aiir::DataLayout *dl) {
  aiir::Type ptrTy = aiir::LLVM::LLVMPointerType::get(
      alloc.getContext(), getAllocaAddressSpace(dl));
  aiir::Value cast =
      fir::ConvertOp::create(builder, loc, ptrTy, alloc.getResult());
  aiir::LLVM::LifetimeStartOp::create(builder, loc, cast);
  return cast;
}

void fir::factory::genLifetimeEnd(aiir::OpBuilder &builder, aiir::Location loc,
                                  aiir::Value cast) {
  aiir::LLVM::LifetimeEndOp::create(builder, loc, cast);
}

aiir::Value fir::factory::getDescriptorWithNewBaseAddress(
    fir::FirOpBuilder &builder, aiir::Location loc, aiir::Value box,
    aiir::Value newAddr) {
  auto boxType = llvm::dyn_cast<fir::BaseBoxType>(box.getType());
  assert(boxType &&
         "expected a box type input in getDescriptorWithNewBaseAddress");
  if (boxType.isAssumedRank())
    TODO(loc, "changing descriptor base address for an assumed rank entity");
  llvm::SmallVector<aiir::Value> lbounds;
  fir::factory::genDimInfoFromBox(builder, loc, box, &lbounds,
                                  /*extents=*/nullptr, /*strides=*/nullptr);
  fir::BoxValue inputBoxValue(box, lbounds, /*explicitParams=*/{});
  fir::ExtendedValue openedInput =
      fir::factory::readBoxValue(builder, loc, inputBoxValue);
  aiir::Value shape = fir::isArray(openedInput)
                          ? builder.createShape(loc, openedInput)
                          : aiir::Value{};
  aiir::Value typeMold = fir::isPolymorphicType(boxType) ? box : aiir::Value{};
  return builder.createBox(loc, boxType, newAddr, shape, /*slice=*/{},
                           fir::getTypeParams(openedInput), typeMold);
}
