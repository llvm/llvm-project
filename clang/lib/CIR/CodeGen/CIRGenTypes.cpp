#include "CIRGenTypes.h"
#include "CIRGenCall.h"
#include "CIRGenFunctionInfo.h"
#include "CIRGenModule.h"
#include "CallingConv.h"
#include "TargetInfo.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/RecordLayout.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace cir;

unsigned CIRGenTypes::ClangCallConvToCIRCallConv(clang::CallingConv CC) {
  assert(CC == CC_C && "No other calling conventions implemented.");
  return cir::CallingConv::C;
}

CIRGenTypes::CIRGenTypes(CIRGenModule &cgm)
    : Context(cgm.getASTContext()), Builder(cgm.getBuilder()), CGM{cgm},
      Target(cgm.getTarget()), TheCXXABI(cgm.getCXXABI()),
      TheABIInfo(cgm.getTargetCIRGenInfo().getABIInfo()) {
  SkippedLayout = false;
}

CIRGenTypes::~CIRGenTypes() {
  for (llvm::FoldingSet<CIRGenFunctionInfo>::iterator I = FunctionInfos.begin(),
                                                      E = FunctionInfos.end();
       I != E;)
    delete &*I++;
}

// This is CIR's version of CIRGenTypes::addRecordTypeName
std::string CIRGenTypes::getRecordTypeName(const clang::RecordDecl *recordDecl,
                                           StringRef suffix) {
  llvm::SmallString<256> typeName;
  llvm::raw_svector_ostream outStream(typeName);

  PrintingPolicy policy = recordDecl->getASTContext().getPrintingPolicy();
  policy.SuppressInlineNamespace = false;

  if (recordDecl->getIdentifier()) {
    if (recordDecl->getDeclContext())
      recordDecl->printQualifiedName(outStream, policy);
    else
      recordDecl->printName(outStream, policy);
  } else if (auto *typedefNameDecl = recordDecl->getTypedefNameForAnonDecl()) {
    if (typedefNameDecl->getDeclContext())
      typedefNameDecl->printQualifiedName(outStream, policy);
    else
      typedefNameDecl->printName(outStream);
  } else {
    outStream << Builder.getUniqueAnonRecordName();
  }

  if (!suffix.empty())
    outStream << suffix;

  return std::string(typeName);
}

/// Return true if the specified type is already completely laid out.
bool CIRGenTypes::isRecordLayoutComplete(const Type *Ty) const {
  llvm::DenseMap<const Type *, mlir::cir::StructType>::const_iterator I =
      recordDeclTypes.find(Ty);
  return I != recordDeclTypes.end() && I->second.getBody();
}

static bool
isSafeToConvert(QualType T, CIRGenTypes &CGT,
                llvm::SmallPtrSet<const RecordDecl *, 16> &AlreadyChecked);

/// Return true if it is safe to convert the specified record decl to IR and lay
/// it out, false if doing so would cause us to get into a recursive compilation
/// mess.
static bool
isSafeToConvert(const RecordDecl *RD, CIRGenTypes &CGT,
                llvm::SmallPtrSet<const RecordDecl *, 16> &AlreadyChecked) {
  // If we have already checked this type (maybe the same type is used by-value
  // multiple times in multiple structure fields, don't check again.
  if (!AlreadyChecked.insert(RD).second)
    return true;

  const Type *Key = CGT.getContext().getTagDeclType(RD).getTypePtr();

  // If this type is already laid out, converting it is a noop.
  if (CGT.isRecordLayoutComplete(Key))
    return true;

  // If this type is currently being laid out, we can't recursively compile it.
  if (CGT.isRecordBeingLaidOut(Key))
    return false;

  // If this type would require laying out bases that are currently being laid
  // out, don't do it.  This includes virtual base classes which get laid out
  // when a class is translated, even though they aren't embedded by-value into
  // the class.
  if (const CXXRecordDecl *CRD = dyn_cast<CXXRecordDecl>(RD)) {
    for (const auto &I : CRD->bases())
      if (!isSafeToConvert(I.getType()->castAs<RecordType>()->getDecl(), CGT,
                           AlreadyChecked))
        return false;
  }

  // If this type would require laying out members that are currently being laid
  // out, don't do it.
  for (const auto *I : RD->fields())
    if (!isSafeToConvert(I->getType(), CGT, AlreadyChecked))
      return false;

  // If there are no problems, lets do it.
  return true;
}

/// Return true if it is safe to convert this field type, which requires the
/// structure elements contained by-value to all be recursively safe to convert.
static bool
isSafeToConvert(QualType T, CIRGenTypes &CGT,
                llvm::SmallPtrSet<const RecordDecl *, 16> &AlreadyChecked) {
  // Strip off atomic type sugar.
  if (const auto *AT = T->getAs<AtomicType>())
    T = AT->getValueType();

  // If this is a record, check it.
  if (const auto *RT = T->getAs<RecordType>())
    return isSafeToConvert(RT->getDecl(), CGT, AlreadyChecked);

  // If this is an array, check the elements, which are embedded inline.
  if (const auto *AT = CGT.getContext().getAsArrayType(T))
    return isSafeToConvert(AT->getElementType(), CGT, AlreadyChecked);

  // Otherwise, there is no concern about transforming this. We only care about
  // things that are contained by-value in a structure that can have another
  // structure as a member.
  return true;
}

// Return true if it is safe to convert the specified record decl to CIR and lay
// it out, false if doing so would cause us to get into a recursive compilation
// mess.
static bool isSafeToConvert(const RecordDecl *RD, CIRGenTypes &CGT) {
  // If no structs are being laid out, we can certainly do this one.
  if (CGT.noRecordsBeingLaidOut())
    return true;

  llvm::SmallPtrSet<const RecordDecl *, 16> AlreadyChecked;
  return isSafeToConvert(RD, CGT, AlreadyChecked);
}

/// Lay out a tagged decl type like struct or union.
mlir::Type CIRGenTypes::convertRecordDeclType(const clang::RecordDecl *RD) {
  // TagDecl's are not necessarily unique, instead use the (clang) type
  // connected to the decl.
  const auto *key = Context.getTagDeclType(RD).getTypePtr();
  mlir::cir::StructType entry = recordDeclTypes[key];

  // Handle forward decl / incomplete types.
  if (!entry) {
    auto name = getRecordTypeName(RD, "");
    entry = Builder.getStructTy({}, name, /*body=*/false, /*packed=*/false, RD);
    recordDeclTypes[key] = entry;
  }

  RD = RD->getDefinition();
  if (!RD || !RD->isCompleteDefinition() || entry.getBody())
    return entry;

  // If converting this type would cause us to infinitely loop, don't do it!
  if (!isSafeToConvert(RD, *this)) {
    DeferredRecords.push_back(RD);
    return entry;
  }

  // Okay, this is a definition of a type. Compile the implementation now.
  bool InsertResult = RecordsBeingLaidOut.insert(key).second;
  (void)InsertResult;
  assert(InsertResult && "Recursively compiling a struct?");

  // Force conversion of non-virtual base classes recursively.
  if (const auto *cxxRecordDecl = dyn_cast<CXXRecordDecl>(RD)) {
    for (const auto &I : cxxRecordDecl->bases()) {
      if (I.isVirtual())
        continue;
      convertRecordDeclType(I.getType()->castAs<RecordType>()->getDecl());
    }
  }

  // Layout fields.
  std::unique_ptr<CIRGenRecordLayout> Layout = computeRecordLayout(RD, &entry);
  recordDeclTypes[key] = entry;
  CIRGenRecordLayouts[key] = std::move(Layout);

  // We're done laying out this struct.
  bool EraseResult = RecordsBeingLaidOut.erase(key);
  (void)EraseResult;
  assert(EraseResult && "struct not in RecordsBeingLaidOut set?");

  // If this struct blocked a FunctionType conversion, then recompute whatever
  // was derived from that.
  // FIXME: This is hugely overconservative.
  if (SkippedLayout)
    TypeCache.clear();

  // If we're done converting the outer-most record, then convert any deferred
  // structs as well.
  if (RecordsBeingLaidOut.empty())
    while (!DeferredRecords.empty())
      convertRecordDeclType(DeferredRecords.pop_back_val());

  return entry;
}

mlir::Type CIRGenTypes::convertTypeForMem(clang::QualType qualType,
                                          bool forBitField) {
  assert(!qualType->isConstantMatrixType() && "Matrix types NYI");

  mlir::Type convertedType = ConvertType(qualType);

  assert(!forBitField && "Bit fields NYI");
  assert(!qualType->isBitIntType() && "BitIntType NYI");

  return convertedType;
}

mlir::MLIRContext &CIRGenTypes::getMLIRContext() const {
  return *Builder.getContext();
}

mlir::Type CIRGenTypes::ConvertFunctionTypeInternal(QualType QFT) {
  assert(QFT.isCanonical());
  const Type *Ty = QFT.getTypePtr();
  const FunctionType *FT = cast<FunctionType>(QFT.getTypePtr());
  // First, check whether we can build the full fucntion type. If the function
  // type depends on an incomplete type (e.g. a struct or enum), we cannot lower
  // the function type.
  assert(isFuncTypeConvertible(FT) && "NYI");

  // While we're converting the parameter types for a function, we don't want to
  // recursively convert any pointed-to structs. Converting directly-used
  // structs is ok though.
  assert(RecordsBeingLaidOut.insert(Ty).second && "NYI");

  // The function type can be built; call the appropriate routines to build it
  const CIRGenFunctionInfo *FI;
  if (const auto *FPT = dyn_cast<FunctionProtoType>(FT)) {
    FI = &arrangeFreeFunctionType(
        CanQual<FunctionProtoType>::CreateUnsafe(QualType(FPT, 0)));
  } else {
    const FunctionNoProtoType *FNPT = cast<FunctionNoProtoType>(FT);
    FI = &arrangeFreeFunctionType(
        CanQual<FunctionNoProtoType>::CreateUnsafe(QualType(FNPT, 0)));
  }

  mlir::Type ResultType = nullptr;
  // If there is something higher level prodding our CIRGenFunctionInfo, then
  // don't recurse into it again.
  assert(!FunctionsBeingProcessed.count(FI) && "NYI");

  // Otherwise, we're good to go, go ahead and convert it.
  ResultType = GetFunctionType(*FI);

  RecordsBeingLaidOut.erase(Ty);

  assert(!SkippedLayout && "Shouldn't have skipped anything yet");

  if (RecordsBeingLaidOut.empty())
    while (!DeferredRecords.empty())
      convertRecordDeclType(DeferredRecords.pop_back_val());

  return ResultType;
}

/// Return true if the specified type in a function parameter or result position
/// can be converted to a CIR type at this point. This boils down to being
/// whether it is complete, as well as whether we've temporarily deferred
/// expanding the type because we're in a recursive context.
bool CIRGenTypes::isFuncParamTypeConvertible(clang::QualType Ty) {
  // Some ABIs cannot have their member pointers represented in LLVM IR unless
  // certain circumstances have been reached.
  assert(!Ty->getAs<MemberPointerType>() && "NYI");

  // If this isn't a tagged type, we can convert it!
  const TagType *TT = Ty->getAs<TagType>();
  if (!TT)
    return true;

  // Incomplete types cannot be converted.
  if (TT->isIncompleteType())
    return false;

  // If this is an enum, then it is always safe to convert.
  const RecordType *RT = dyn_cast<RecordType>(TT);
  if (!RT)
    return true;

  // Otherwise, we have to be careful.  If it is a struct that we're in the
  // process of expanding, then we can't convert the function type.  That's ok
  // though because we must be in a pointer context under the struct, so we can
  // just convert it to a dummy type.
  //
  // We decide this by checking whether ConvertRecordDeclType returns us an
  // opaque type for a struct that we know is defined.
  return isSafeToConvert(RT->getDecl(), *this);
}

/// Code to verify a given function type is complete, i.e. the return type and
/// all of the parameter types are complete. Also check to see if we are in a
/// RS_StructPointer context, and if so whether any struct types have been
/// pended. If so, we don't want to ask the ABI lowering code to handle a type
/// that cannot be converted to a CIR type.
bool CIRGenTypes::isFuncTypeConvertible(const FunctionType *FT) {
  if (!isFuncParamTypeConvertible(FT->getReturnType()))
    return false;

  if (const auto *FPT = dyn_cast<FunctionProtoType>(FT))
    for (unsigned i = 0, e = FPT->getNumParams(); i != e; i++)
      if (!isFuncParamTypeConvertible(FPT->getParamType(i)))
        return false;

  return true;
}

/// ConvertType - Convert the specified type to its MLIR form.
mlir::Type CIRGenTypes::ConvertType(QualType T) {
  T = Context.getCanonicalType(T);
  const Type *Ty = T.getTypePtr();

  // For the device-side compilation, CUDA device builtin surface/texture types
  // may be represented in different types.
  assert(!Context.getLangOpts().CUDAIsDevice && "not implemented");

  if (const auto *recordType = dyn_cast<RecordType>(T))
    return convertRecordDeclType(recordType->getDecl());

  // See if type is already cached.
  TypeCacheTy::iterator TCI = TypeCache.find(Ty);
  // If type is found in map then use it. Otherwise, convert type T.
  if (TCI != TypeCache.end())
    return TCI->second;

  // If we don't have it in the cache, convert it now.
  mlir::Type ResultType = nullptr;
  switch (Ty->getTypeClass()) {
  case Type::Record: // Handled above.
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.inc"
    llvm_unreachable("Non-canonical or dependent types aren't possible.");

  case Type::ArrayParameter:
    llvm_unreachable("NYI");

  case Type::Builtin: {
    switch (cast<BuiltinType>(Ty)->getKind()) {
    case BuiltinType::WasmExternRef:
    case BuiltinType::SveBoolx2:
    case BuiltinType::SveBoolx4:
    case BuiltinType::SveCount:
      llvm_unreachable("NYI");
    case BuiltinType::Void:
      // TODO(cir): how should we model this?
      ResultType = CGM.VoidTy;
      break;

    case BuiltinType::ObjCId:
    case BuiltinType::ObjCClass:
    case BuiltinType::ObjCSel:
      // TODO(cir): probably same as BuiltinType::Void
      assert(0 && "not implemented");
      break;

    case BuiltinType::Bool:
      ResultType = ::mlir::cir::BoolType::get(Builder.getContext());
      break;

    // Signed types.
    case BuiltinType::Accum:
    case BuiltinType::Char_S:
    case BuiltinType::Fract:
    case BuiltinType::Int:
    case BuiltinType::Long:
    case BuiltinType::LongAccum:
    case BuiltinType::LongFract:
    case BuiltinType::LongLong:
    case BuiltinType::SChar:
    case BuiltinType::Short:
    case BuiltinType::ShortAccum:
    case BuiltinType::ShortFract:
    case BuiltinType::WChar_S:
    // Saturated signed types.
    case BuiltinType::SatAccum:
    case BuiltinType::SatFract:
    case BuiltinType::SatLongAccum:
    case BuiltinType::SatLongFract:
    case BuiltinType::SatShortAccum:
    case BuiltinType::SatShortFract:
      ResultType =
          mlir::cir::IntType::get(Builder.getContext(), Context.getTypeSize(T),
                                  /*isSigned=*/true);
      break;
    // Unsigned types.
    case BuiltinType::Char16:
    case BuiltinType::Char32:
    case BuiltinType::Char8:
    case BuiltinType::Char_U:
    case BuiltinType::UAccum:
    case BuiltinType::UChar:
    case BuiltinType::UFract:
    case BuiltinType::UInt:
    case BuiltinType::ULong:
    case BuiltinType::ULongAccum:
    case BuiltinType::ULongFract:
    case BuiltinType::ULongLong:
    case BuiltinType::UShort:
    case BuiltinType::UShortAccum:
    case BuiltinType::UShortFract:
    case BuiltinType::WChar_U:
    // Saturated unsigned types.
    case BuiltinType::SatUAccum:
    case BuiltinType::SatUFract:
    case BuiltinType::SatULongAccum:
    case BuiltinType::SatULongFract:
    case BuiltinType::SatUShortAccum:
    case BuiltinType::SatUShortFract:
      ResultType =
          mlir::cir::IntType::get(Builder.getContext(), Context.getTypeSize(T),
                                  /*isSigned=*/false);
      break;

    case BuiltinType::Float16:
      ResultType = Builder.getF16Type();
      break;
    case BuiltinType::Half:
      // Should be the same as above?
      assert(0 && "not implemented");
      break;
    case BuiltinType::BFloat16:
      ResultType = Builder.getBF16Type();
      break;
    case BuiltinType::Float:
      ResultType = CGM.FloatTy;
      break;
    case BuiltinType::Double:
      ResultType = CGM.DoubleTy;
      break;
    case BuiltinType::LongDouble:
      ResultType = Builder.getFloatTyForFormat(Context.getFloatTypeSemantics(T),
                                               /*useNativeHalf=*/false);
      break;
    case BuiltinType::Float128:
    case BuiltinType::Ibm128:
      // FIXME: look at Context.getFloatTypeSemantics(T) and getTypeForFormat
      // on LLVM codegen.
      assert(0 && "not implemented");
      break;

    case BuiltinType::NullPtr:
      // Add proper CIR type for it? this looks mostly useful for sema related
      // things (like for overloads accepting void), for now, given that
      // `sizeof(std::nullptr_t)` is equal to `sizeof(void *)`, model
      // std::nullptr_t as !cir.ptr<!void>
      ResultType = Builder.getVoidPtrTy();
      break;

    case BuiltinType::UInt128:
    case BuiltinType::Int128:
      assert(0 && "not implemented");
      // FIXME: ResultType = Builder.getIntegerType(128);
      break;

#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix)                   \
  case BuiltinType::Id:
#include "clang/Basic/OpenCLImageTypes.def"
#define EXT_OPAQUE_TYPE(ExtType, Id, Ext) case BuiltinType::Id:
#include "clang/Basic/OpenCLExtensionTypes.def"
    case BuiltinType::OCLSampler:
    case BuiltinType::OCLEvent:
    case BuiltinType::OCLClkEvent:
    case BuiltinType::OCLQueue:
    case BuiltinType::OCLReserveID:
      assert(0 && "not implemented");
      break;
    case BuiltinType::SveInt8:
    case BuiltinType::SveUint8:
    case BuiltinType::SveInt8x2:
    case BuiltinType::SveUint8x2:
    case BuiltinType::SveInt8x3:
    case BuiltinType::SveUint8x3:
    case BuiltinType::SveInt8x4:
    case BuiltinType::SveUint8x4:
    case BuiltinType::SveInt16:
    case BuiltinType::SveUint16:
    case BuiltinType::SveInt16x2:
    case BuiltinType::SveUint16x2:
    case BuiltinType::SveInt16x3:
    case BuiltinType::SveUint16x3:
    case BuiltinType::SveInt16x4:
    case BuiltinType::SveUint16x4:
    case BuiltinType::SveInt32:
    case BuiltinType::SveUint32:
    case BuiltinType::SveInt32x2:
    case BuiltinType::SveUint32x2:
    case BuiltinType::SveInt32x3:
    case BuiltinType::SveUint32x3:
    case BuiltinType::SveInt32x4:
    case BuiltinType::SveUint32x4:
    case BuiltinType::SveInt64:
    case BuiltinType::SveUint64:
    case BuiltinType::SveInt64x2:
    case BuiltinType::SveUint64x2:
    case BuiltinType::SveInt64x3:
    case BuiltinType::SveUint64x3:
    case BuiltinType::SveInt64x4:
    case BuiltinType::SveUint64x4:
    case BuiltinType::SveBool:
    case BuiltinType::SveFloat16:
    case BuiltinType::SveFloat16x2:
    case BuiltinType::SveFloat16x3:
    case BuiltinType::SveFloat16x4:
    case BuiltinType::SveFloat32:
    case BuiltinType::SveFloat32x2:
    case BuiltinType::SveFloat32x3:
    case BuiltinType::SveFloat32x4:
    case BuiltinType::SveFloat64:
    case BuiltinType::SveFloat64x2:
    case BuiltinType::SveFloat64x3:
    case BuiltinType::SveFloat64x4:
    case BuiltinType::SveBFloat16:
    case BuiltinType::SveBFloat16x2:
    case BuiltinType::SveBFloat16x3:
    case BuiltinType::SveBFloat16x4: {
      assert(0 && "not implemented");
      break;
    }
#define PPC_VECTOR_TYPE(Name, Id, Size)                                        \
  case BuiltinType::Id:                                                        \
    assert(0 && "not implemented");                                            \
    break;
#include "clang/Basic/PPCTypes.def"
#define RVV_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/RISCVVTypes.def"
      {
        assert(0 && "not implemented");
        break;
      }
    case BuiltinType::Dependent:
#define BUILTIN_TYPE(Id, SingletonId)
#define PLACEHOLDER_TYPE(Id, SingletonId) case BuiltinType::Id:
#include "clang/AST/BuiltinTypes.def"
      llvm_unreachable("Unexpected placeholder builtin type!");
    }
    break;
  }
  case Type::Auto:
  case Type::DeducedTemplateSpecialization:
    llvm_unreachable("Unexpected undeduced type!");
  case Type::Complex: {
    assert(0 && "not implemented");
    break;
  }
  case Type::LValueReference:
  case Type::RValueReference: {
    const ReferenceType *RTy = cast<ReferenceType>(Ty);
    QualType ETy = RTy->getPointeeType();
    auto PointeeType = convertTypeForMem(ETy);
    // TODO(cir): use Context.getTargetAddressSpace(ETy) on pointer
    ResultType =
        ::mlir::cir::PointerType::get(Builder.getContext(), PointeeType);
    assert(ResultType && "Cannot get pointer type?");
    break;
  }
  case Type::Pointer: {
    const PointerType *PTy = cast<PointerType>(Ty);
    QualType ETy = PTy->getPointeeType();
    assert(!ETy->isConstantMatrixType() && "not implemented");

    mlir::Type PointeeType = ConvertType(ETy);

    // Treat effectively as a *i8.
    // if (PointeeType->isVoidTy())
    //  PointeeType = Builder.getI8Type();

    // FIXME: add address specifier to cir::PointerType?
    ResultType =
        ::mlir::cir::PointerType::get(Builder.getContext(), PointeeType);
    assert(ResultType && "Cannot get pointer type?");
    break;
  }

  case Type::VariableArray: {
    assert(0 && "not implemented");
    break;
  }
  case Type::IncompleteArray: {
    assert(0 && "not implemented");
    break;
  }
  case Type::ConstantArray: {
    const ConstantArrayType *A = cast<ConstantArrayType>(Ty);
    auto EltTy = convertTypeForMem(A->getElementType());

    // FIXME(cir): add a `isSized` method to CIRGenBuilder.
    auto isSized = [&](mlir::Type ty) {
      if (ty.isIntOrFloat() ||
          ty.isa<mlir::cir::PointerType, mlir::cir::StructType,
                 mlir::cir::ArrayType, mlir::cir::BoolType,
                 mlir::cir::IntType>())
        return true;
      assert(0 && "not implemented");
      return false;
    };

    // FIXME: In LLVM, "lower arrays of undefined struct type to arrays of
    // i8 just to have a concrete type". Not sure this makes sense in CIR yet.
    assert(isSized(EltTy) && "not implemented");
    ResultType = ::mlir::cir::ArrayType::get(Builder.getContext(), EltTy,
                                             A->getSize().getZExtValue());
    break;
  }
  case Type::ExtVector:
  case Type::Vector: {
    assert(0 && "not implemented");
    break;
  }
  case Type::ConstantMatrix: {
    assert(0 && "not implemented");
    break;
  }
  case Type::FunctionNoProto:
  case Type::FunctionProto:
    ResultType = ConvertFunctionTypeInternal(T);
    break;
  case Type::ObjCObject:
    assert(0 && "not implemented");
    break;

  case Type::ObjCInterface: {
    assert(0 && "not implemented");
    break;
  }

  case Type::ObjCObjectPointer: {
    assert(0 && "not implemented");
    break;
  }

  case Type::Enum: {
    const EnumDecl *ED = cast<EnumType>(Ty)->getDecl();
    if (ED->isCompleteDefinition() || ED->isFixed())
      return ConvertType(ED->getIntegerType());
    // Return a placeholder 'i32' type.  This can be changed later when the
    // type is defined (see UpdateCompletedType), but is likely to be the
    // "right" answer.
    ResultType = CGM.UInt32Ty;
    break;
  }

  case Type::BlockPointer: {
    assert(0 && "not implemented");
    break;
  }

  case Type::MemberPointer: {
    assert(0 && "not implemented");
    break;
  }

  case Type::Atomic: {
    QualType valueType = cast<AtomicType>(Ty)->getValueType();
    ResultType = convertTypeForMem(valueType);

    // Pad out to the inflated size if necessary.
    uint64_t valueSize = Context.getTypeSize(valueType);
    uint64_t atomicSize = Context.getTypeSize(Ty);
    if (valueSize != atomicSize) {
      llvm_unreachable("NYI");
    }
    break;
  }
  case Type::Pipe: {
    assert(0 && "not implemented");
    break;
  }
  case Type::BitInt: {
    assert(0 && "not implemented");
    break;
  }
  }

  assert(ResultType && "Didn't convert a type?");

  TypeCache[Ty] = ResultType;
  return ResultType;
}

const CIRGenFunctionInfo &CIRGenTypes::arrangeCIRFunctionInfo(
    CanQualType resultType, bool instanceMethod, bool chainCall,
    llvm::ArrayRef<CanQualType> argTypes, FunctionType::ExtInfo info,
    llvm::ArrayRef<FunctionProtoType::ExtParameterInfo> paramInfos,
    RequiredArgs required) {
  assert(llvm::all_of(argTypes,
                      [](CanQualType T) { return T.isCanonicalAsParam(); }));

  // Lookup or create unique function info.
  llvm::FoldingSetNodeID ID;
  CIRGenFunctionInfo::Profile(ID, instanceMethod, chainCall, info, paramInfos,
                              required, resultType, argTypes);

  void *insertPos = nullptr;
  CIRGenFunctionInfo *FI = FunctionInfos.FindNodeOrInsertPos(ID, insertPos);
  if (FI)
    return *FI;

  unsigned CC = ClangCallConvToCIRCallConv(info.getCC());

  // Construction the function info. We co-allocate the ArgInfos.
  FI = CIRGenFunctionInfo::create(CC, instanceMethod, chainCall, info,
                                  paramInfos, resultType, argTypes, required);
  FunctionInfos.InsertNode(FI, insertPos);

  bool inserted = FunctionsBeingProcessed.insert(FI).second;
  (void)inserted;
  assert(inserted && "Recursively being processed?");

  // Compute ABI inforamtion.
  assert(info.getCC() != clang::CallingConv::CC_SpirFunction && "NYI");
  assert(info.getCC() != CC_Swift && info.getCC() != CC_SwiftAsync &&
         "Swift NYI");
  getABIInfo().computeInfo(*FI);

  // Loop over all of the computed argument and return value info. If any of
  // them are direct or extend without a specified coerce type, specify the
  // default now.
  ABIArgInfo &retInfo = FI->getReturnInfo();
  if (retInfo.canHaveCoerceToType() && retInfo.getCoerceToType() == nullptr)
    retInfo.setCoerceToType(ConvertType(FI->getReturnType()));

  for (auto &I : FI->arguments())
    if (I.info.canHaveCoerceToType() && I.info.getCoerceToType() == nullptr)
      I.info.setCoerceToType(ConvertType(I.type));

  bool erased = FunctionsBeingProcessed.erase(FI);
  (void)erased;
  assert(erased && "Not in set?");

  return *FI;
}

const CIRGenFunctionInfo &CIRGenTypes::arrangeGlobalDeclaration(GlobalDecl GD) {
  assert(!dyn_cast<ObjCMethodDecl>(GD.getDecl()) &&
         "This is reported as a FIXME in LLVM codegen");
  const auto *FD = cast<FunctionDecl>(GD.getDecl());

  if (isa<CXXConstructorDecl>(GD.getDecl()) ||
      isa<CXXDestructorDecl>(GD.getDecl()))
    return arrangeCXXStructorDeclaration(GD);

  return arrangeFunctionDeclaration(FD);
}

// UpdateCompletedType - When we find the full definition for a TagDecl,
// replace the 'opaque' type we previously made for it if applicable.
void CIRGenTypes::UpdateCompletedType(const TagDecl *TD) {
  // If this is an enum being completed, then we flush all non-struct types
  // from the cache. This allows function types and other things that may be
  // derived from the enum to be recomputed.
  if (const auto *ED = dyn_cast<EnumDecl>(TD)) {
    // Only flush the cache if we've actually already converted this type.
    if (TypeCache.count(ED->getTypeForDecl())) {
      // Okay, we formed some types based on this.  We speculated that the enum
      // would be lowered to i32, so we only need to flush the cache if this
      // didn't happen.
      if (!ConvertType(ED->getIntegerType()).isInteger(32))
        TypeCache.clear();
    }
    // If necessary, provide the full definition of a type only used with a
    // declaration so far.
    assert(!UnimplementedFeature::generateDebugInfo());
    return;
  }

  // If we completed a RecordDecl that we previously used and converted to an
  // anonymous type, then go ahead and complete it now.
  const auto *RD = cast<RecordDecl>(TD);
  if (RD->isDependentType())
    return;

  // Only complete if we converted it already. If we haven't converted it yet,
  // we'll just do it lazily.
  if (recordDeclTypes.count(Context.getTagDeclType(RD).getTypePtr()))
    convertRecordDeclType(RD);

  // If necessary, provide the full definition of a type only used with a
  // declaration so far.
  if (CGM.getModuleDebugInfo())
    llvm_unreachable("NYI");
}

/// getCIRGenRecordLayout - Return record layout info for the given record decl.
const CIRGenRecordLayout &
CIRGenTypes::getCIRGenRecordLayout(const RecordDecl *RD) {
  const auto *Key = Context.getTagDeclType(RD).getTypePtr();

  auto I = CIRGenRecordLayouts.find(Key);
  if (I != CIRGenRecordLayouts.end())
    return *I->second;

  // Compute the type information.
  convertRecordDeclType(RD);

  // Now try again.
  I = CIRGenRecordLayouts.find(Key);

  assert(I != CIRGenRecordLayouts.end() &&
         "Unable to find record layout information for type");
  return *I->second;
}

bool CIRGenTypes::isZeroInitializable(QualType T) {
  if (T->getAs<PointerType>())
    return Context.getTargetNullPointerValue(T) == 0;

  if (const auto *AT = Context.getAsArrayType(T)) {
    if (isa<IncompleteArrayType>(AT))
      return true;
    if (const auto *CAT = dyn_cast<ConstantArrayType>(AT))
      if (Context.getConstantArrayElementCount(CAT) == 0)
        return true;
    T = Context.getBaseElementType(T);
  }

  // Records are non-zero-initializable if they contain any
  // non-zero-initializable subobjects.
  if (const RecordType *RT = T->getAs<RecordType>()) {
    const RecordDecl *RD = RT->getDecl();
    return isZeroInitializable(RD);
  }

  // We have to ask the ABI about member pointers.
  if (const MemberPointerType *MPT = T->getAs<MemberPointerType>())
    llvm_unreachable("NYI");

  // Everything else is okay.
  return true;
}

bool CIRGenTypes::isZeroInitializable(const RecordDecl *RD) {
  return getCIRGenRecordLayout(RD).isZeroInitializable();
}
