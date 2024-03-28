//===- AddDebugFoundation.cpp -- add basic debug linetable info -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// This pass populates some debug information for the module and functions.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Frontend/Debug/Options.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace fir {
#define GEN_PASS_DEF_ADDDEBUGFOUNDATION
#define GEN_PASS_DECL_ADDDEBUGFOUNDATION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-add-debug-foundation"

namespace {

class AddDebugFoundationPass
    : public fir::impl::AddDebugFoundationBase<AddDebugFoundationPass> {
public:
  AddDebugFoundationPass(unsigned level) { debugLevel = level; }
  AddDebugFoundationPass() {
    debugLevel = static_cast<unsigned>(llvm::codegenoptions::FullDebugInfo);
  }
  void runOnOperation() override;

private:
  void handleGlobalOp(fir::GlobalOp glocalOp, mlir::LLVM::DIFileAttr fileAttr,
                      mlir::LLVM::DIScopeAttr scope);
  void handleFunctionOp(mlir::func::FuncOp funcOp,
                        mlir::LLVM::DIFileAttr fileAttr,
                        mlir::LLVM::DICompileUnitAttr cuAttr,
                        llvm::StringRef parentFilePath);
  void handleDeclareOp(fir::DeclareOp declOp, mlir::func::FuncOp funcOp,
                       mlir::LLVM::DIFileAttr fileAttr,
                       mlir::LLVM::DIScopeAttr scopeAttr, uint32_t &argNo);
};

class TypeConverter {
  mlir::LLVM::DITypeAttr convertCharacterType(mlir::MLIRContext *context,
                                              fir::CharacterType Ty,
                                              mlir::LLVM::DIFileAttr fileAttr,
                                              mlir::LLVM::DIScopeAttr scope,
                                              mlir::Location loc);

  mlir::LLVM::DITypeAttr convertRecordType(mlir::MLIRContext *context,
                                           fir::RecordType Ty,
                                           mlir::LLVM::DIFileAttr fileAttr,
                                           mlir::LLVM::DIScopeAttr scope,
                                           mlir::Location loc);

  mlir::LLVM::DITypeAttr convertSequenceType(mlir::MLIRContext *context,
                                             fir::SequenceType Ty,
                                             mlir::LLVM::DIFileAttr fileAttr,
                                             mlir::LLVM::DIScopeAttr scope,
                                             mlir::Location loc);

public:
  TypeConverter(mlir::ModuleOp m) : module(m) {}

  mlir::LLVM::DITypeAttr convert(mlir::MLIRContext *context, mlir::Type Ty,
                                 mlir::LLVM::DIFileAttr fileAttr,
                                 mlir::LLVM::DIScopeAttr scope,
                                 mlir::Location loc);

private:
  mlir::ModuleOp module;
};

static mlir::LLVM::DIEmissionKind getEmissionKind(unsigned debugLevel) {
  switch (debugLevel) {
  case llvm::codegenoptions::NoDebugInfo:
  case llvm::codegenoptions::LocTrackingOnly:
    return mlir::LLVM::DIEmissionKind::None;
  case llvm::codegenoptions::DebugLineTablesOnly:
    return mlir::LLVM::DIEmissionKind::LineTablesOnly;
  case llvm::codegenoptions::DebugDirectivesOnly:
  case llvm::codegenoptions::DebugInfoConstructor:
  case llvm::codegenoptions::LimitedDebugInfo:
  case llvm::codegenoptions::FullDebugInfo:
  case llvm::codegenoptions::UnusedTypeInfo:
    return mlir::LLVM::DIEmissionKind::Full;
    break;
  default:
    assert(false && "Unhandled debug level!");
  }
}

static mlir::LLVM::DITypeAttr genPlaceholderType(mlir::MLIRContext *context) {
  return mlir::LLVM::DIBasicTypeAttr::get(
      context, llvm::dwarf::DW_TAG_base_type, "void", 32, 1);
}

static mlir::LLVM::DITypeAttr genBasicType(mlir::MLIRContext *context,
                                           mlir::StringAttr name,
                                           unsigned bitSize,
                                           unsigned decoding) {
  return mlir::LLVM::DIBasicTypeAttr::get(
      context, llvm::dwarf::DW_TAG_base_type, name, bitSize, decoding);
}

static uint32_t getLineFromLoc(mlir::Location loc) {
  uint32_t line = 1;
  if (auto fileLoc = loc.dyn_cast<mlir::FileLineColLoc>())
    line = fileLoc.getLine();
  return line;
}

static uint32_t getTypeSize(mlir::LLVM::DITypeAttr Ty, mlir::Location loc) {
  if (auto MT = Ty.dyn_cast_or_null<mlir::LLVM::DIBasicTypeAttr>())
    return MT.getSizeInBits();
  else if (auto MT = Ty.dyn_cast_or_null<mlir::LLVM::DIDerivedTypeAttr>())
    return MT.getSizeInBits();
  else if (auto MT = Ty.dyn_cast_or_null<mlir::LLVM::DICompositeTypeAttr>())
    return MT.getSizeInBits();
  TODO(loc, "Unsupported type!");
}

} // namespace

mlir::LLVM::DITypeAttr TypeConverter::convertCharacterType(
    mlir::MLIRContext *context, fir::CharacterType Ty,
    mlir::LLVM::DIFileAttr fileAttr, mlir::LLVM::DIScopeAttr scope,
    mlir::Location loc) {

  if (!Ty.hasConstantLen())
    return genPlaceholderType(context);

  fir::CharacterType::LenType len = Ty.getLen();
  auto charTy =
      genBasicType(context, mlir::StringAttr::get(context, Ty.getMnemonic()),
                   Ty.getFKind() * 8, llvm::dwarf::DW_ATE_unsigned_char);

  if (len == 1)
    return charTy;

  auto intTy = mlir::IntegerType::get(context, 64);
  auto countAttr = mlir::IntegerAttr::get(intTy, llvm::APInt(64, len));
  auto lowerAttr = mlir::IntegerAttr::get(intTy, llvm::APInt(64, 1));
  auto subrangeTy = mlir::LLVM::DISubrangeAttr::get(
      context, countAttr, lowerAttr, nullptr, nullptr);

  return mlir::LLVM::DICompositeTypeAttr::get(
      context, llvm::dwarf::DW_TAG_array_type,
      mlir::StringAttr::get(context, ""), fileAttr, getLineFromLoc(loc), scope,
      charTy, mlir::LLVM::DIFlags::Zero, len * getTypeSize(charTy, loc),
      /*alignInBits*/ 0, {subrangeTy});
}

mlir::LLVM::DITypeAttr
TypeConverter::convertRecordType(mlir::MLIRContext *context, fir::RecordType Ty,
                                 mlir::LLVM::DIFileAttr fileAttr,
                                 mlir::LLVM::DIScopeAttr scope,
                                 mlir::Location loc) {

  llvm::SmallVector<mlir::LLVM::DINodeAttr> elements;
  // The offset for the fields is being calculated by adding
  // the bit sizes of all the previous fields.
  uint64_t offset = 0;

  for (auto CT : Ty.getTypeList()) {
    auto MT =
        convert(context, fir::unwrapRefType(CT.second), fileAttr, scope, loc);
    uint64_t BitSize = getTypeSize(MT, loc);
    auto DT = mlir::LLVM::DIDerivedTypeAttr::get(
        context, llvm::dwarf::DW_TAG_member,
        mlir::StringAttr::get(context, CT.first), MT, BitSize,
        /*alignInBits*/ 0, offset);
    elements.push_back(DT);
    offset += BitSize;
  }

  // TODO: Handle parent type
  mlir::LLVM::DITypeAttr parentTy = mlir::LLVM::DINullTypeAttr::get(context);

  // TODO: The RecordType and FieldType does not seem to have location. So
  // at the moment, we are forced to use whatever location the variable was
  // declared.
  return mlir::LLVM::DICompositeTypeAttr::get(
      context, llvm::dwarf::DW_TAG_structure_type,
      mlir::StringAttr::get(context, Ty.getName()), fileAttr,
      getLineFromLoc(loc), scope, parentTy, mlir::LLVM::DIFlags::Zero, offset,
      /*alignInBits*/ 0, elements);
}

mlir::LLVM::DITypeAttr TypeConverter::convertSequenceType(
    mlir::MLIRContext *context, fir::SequenceType seqTy,
    mlir::LLVM::DIFileAttr fileAttr, mlir::LLVM::DIScopeAttr scope,
    mlir::Location loc) {

  // TODO: Only fixed sizes arrays handled at the moment.
  if (seqTy.hasDynamicExtents())
    return genPlaceholderType(context);

  llvm::SmallVector<mlir::LLVM::DINodeAttr> elements;
  uint64_t size = 0;
  auto elemTy = convert(context, seqTy.getEleTy(), fileAttr, scope, loc);
  for (auto dim : seqTy.getShape()) {
    size += dim;
    auto intTy = mlir::IntegerType::get(context, 64);
    // TODO: Only supporting lower bound of 1 at the moment.
    auto countAttr = mlir::IntegerAttr::get(intTy, llvm::APInt(64, dim));
    auto lowerAttr = mlir::IntegerAttr::get(intTy, llvm::APInt(64, 1));
    auto subrangeTy = mlir::LLVM::DISubrangeAttr::get(
        context, countAttr, lowerAttr, nullptr, nullptr);
    elements.push_back(subrangeTy);
  }
  return mlir::LLVM::DICompositeTypeAttr::get(
      context, llvm::dwarf::DW_TAG_array_type,
      mlir::StringAttr::get(context, ""), fileAttr, getLineFromLoc(loc), scope,
      elemTy, mlir::LLVM::DIFlags::Zero, size * getTypeSize(elemTy, loc),
      /*alignInBits*/ 0, elements);
}

mlir::LLVM::DITypeAttr TypeConverter::convert(mlir::MLIRContext *context,
                                              mlir::Type Ty,
                                              mlir::LLVM::DIFileAttr fileAttr,
                                              mlir::LLVM::DIScopeAttr scope,
                                              mlir::Location loc) {
  fir::KindMapping kindMap = fir::getKindMapping(module);
  if (Ty.isIntOrIndex()) {
    return genBasicType(context, mlir::StringAttr::get(context, "integer"),
                        Ty.getIntOrFloatBitWidth(), llvm::dwarf::DW_ATE_signed);
  } else if (Ty.isa<mlir::FloatType>() || Ty.isa<fir::RealType>()) {
    return genBasicType(context, mlir::StringAttr::get(context, "real"),
                        Ty.getIntOrFloatBitWidth(), llvm::dwarf::DW_ATE_float);
  } else if (auto logTy = Ty.dyn_cast_or_null<fir::LogicalType>()) {
    return genBasicType(context,
                        mlir::StringAttr::get(context, logTy.getMnemonic()),
                        kindMap.getLogicalBitsize(logTy.getFKind()),
                        llvm::dwarf::DW_ATE_boolean);
  } else if (fir::isa_complex(Ty)) {
    unsigned bitWidth;
    if (auto cplxTy = mlir::dyn_cast_or_null<mlir::ComplexType>(Ty)) {
      auto floatTy = cplxTy.getElementType().cast<mlir::FloatType>();
      bitWidth = floatTy.getWidth();
    } else if (auto cplxTy = mlir::dyn_cast_or_null<fir::ComplexType>(Ty)) {
      bitWidth = kindMap.getRealBitsize(cplxTy.getFKind());
    }
    return genBasicType(context, mlir::StringAttr::get(context, "complex"),
                        bitWidth * 2, llvm::dwarf::DW_ATE_complex_float);
  } else if (auto charTy = Ty.dyn_cast_or_null<fir::CharacterType>()) {
    return convertCharacterType(context, charTy, fileAttr, scope, loc);
  } else if (auto recTy = Ty.dyn_cast_or_null<fir::RecordType>()) {
    return convertRecordType(context, recTy, fileAttr, scope, loc);
  } else if (auto seqTy = Ty.dyn_cast_or_null<fir::SequenceType>()) {
    return convertSequenceType(context, seqTy, fileAttr, scope, loc);
  } else if (Ty.isa<fir::PointerType>() || Ty.isa<fir::ReferenceType>() ||
             Ty.isa<fir::BoxType>() || Ty.isa<mlir::TupleType>()) {
    // TODO: These types are currently unhandled. We are generating a
    // placeholder type to allow us to test supported bits.
    return genPlaceholderType(context);
  } else
    TODO(loc, "Unsupported Type!");
}

void AddDebugFoundationPass::handleGlobalOp(fir::GlobalOp globalOp,
                                            mlir::LLVM::DIFileAttr fileAttr,
                                            mlir::LLVM::DIScopeAttr scope) {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *context = &getContext();
  TypeConverter tyConverter(module);

  auto result = fir::NameUniquer::deconstruct(globalOp.getSymName());
  // TODO: Use module information from the 'result' if available
  auto diType =
      tyConverter.convert(context, fir::unwrapRefType(globalOp.getType()),
                          fileAttr, scope, globalOp.getLoc());

  auto gvAttr = mlir::LLVM::DIGlobalVariableAttr::get(
      context, scope, mlir::StringAttr::get(context, result.second.name),
      mlir::StringAttr::get(context, globalOp.getName()), fileAttr,
      getLineFromLoc(globalOp.getLoc()), diType, /*isLocalToUnit*/ true,
      /*isDefinition*/ true, /* alignInBits*/ 0);
  globalOp->setAttr("debug", gvAttr);
}

void AddDebugFoundationPass::handleDeclareOp(fir::DeclareOp declOp,
                                             mlir::func::FuncOp funcOp,
                                             mlir::LLVM::DIFileAttr fileAttr,
                                             mlir::LLVM::DIScopeAttr scopeAttr,
                                             uint32_t &argNo) {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *context = &getContext();
  TypeConverter tyConverter(module);

  auto refOp = declOp.getMemref();
  bool isLocal = refOp.getDefiningOp();
  auto diType =
      tyConverter.convert(context, fir::unwrapRefType(declOp.getType()),
                          fileAttr, scopeAttr, declOp.getLoc());
  auto result = fir::NameUniquer::deconstruct(declOp.getUniqName());
  auto localVarAttr = mlir::LLVM::DILocalVariableAttr::get(
      context, scopeAttr, mlir::StringAttr::get(context, result.second.name),
      fileAttr, getLineFromLoc(declOp.getLoc()), isLocal ? 0 : argNo++,
      /* alignInBits*/ 0, diType);

  if (isLocal)
    refOp.getDefiningOp()->setAttr("debug", localVarAttr);
  else {
    if (auto arg = mlir::dyn_cast_or_null<mlir::BlockArgument>(refOp)) {
      bool done = false;
      // find the LoadOp that loads the block argument and attach local
      // variable attribute to it.
      funcOp.walk([&](fir::LoadOp loadOp) {
        if (done)
          return;
        if (loadOp.getMemref() == declOp) {
          done = true;
          loadOp->setAttr("debug", localVarAttr);
        }
      });
    }
  }
}

void AddDebugFoundationPass::handleFunctionOp(
    mlir::func::FuncOp funcOp, mlir::LLVM::DIFileAttr fileAttr,
    mlir::LLVM::DICompileUnitAttr cuAttr, llvm::StringRef parentFilePath) {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *context = &getContext();
  TypeConverter tyConverter(module);
  mlir::OpBuilder builder(context);

  mlir::Location l = funcOp->getLoc();
  // If fused location has already been created then nothing to do
  // Otherwise, create a fused location.
  if (l.dyn_cast<mlir::FusedLoc>())
    return;

  llvm::StringRef funcFilePath{parentFilePath};
  unsigned funcLine = 1;
  if (auto funcLoc = l.dyn_cast<mlir::FileLineColLoc>()) {
    funcLine = funcLoc.getLine();
    funcFilePath = funcLoc.getFilename().getValue();
  }

  mlir::StringAttr funcName = mlir::StringAttr::get(context, funcOp.getName());
  llvm::SmallVector<mlir::LLVM::DITypeAttr> types;
  for (auto resTy : funcOp.getResultTypes()) {
    auto tyAttr =
        tyConverter.convert(context, resTy, fileAttr, cuAttr, funcOp.getLoc());
    types.push_back(tyAttr);
  }
  for (auto inTy : funcOp.getArgumentTypes()) {
    auto tyAttr =
        tyConverter.convert(context, inTy, fileAttr, cuAttr, funcOp.getLoc());
    types.push_back(tyAttr);
  }
  mlir::LLVM::DISubroutineTypeAttr subTypeAttr =
      mlir::LLVM::DISubroutineTypeAttr::get(
          context, llvm::dwarf::getCallingConvention("DW_CC_normal"), types);

  mlir::LLVM::DIFileAttr funcFileAttr = mlir::LLVM::DIFileAttr::get(
      context, llvm::sys::path::filename(funcFilePath),
      llvm::sys::path::parent_path(funcFilePath));

  // Only definitions need a distinct identifier and a compilation unit.
  mlir::DistinctAttr id;
  mlir::LLVM::DICompileUnitAttr compilationUnit;
  auto subprogramFlags = mlir::LLVM::DISubprogramFlags::Optimized;
  if (!funcOp.isExternal()) {
    id = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
    compilationUnit = cuAttr;
    subprogramFlags =
        subprogramFlags | mlir::LLVM::DISubprogramFlags::Definition;
  }
  auto spAttr = mlir::LLVM::DISubprogramAttr::get(
      context, id, compilationUnit, fileAttr, funcName, funcName, funcFileAttr,
      /*line=*/funcLine, /*scopeline=*/funcLine, subprogramFlags, subTypeAttr);
  funcOp->setLoc(builder.getFusedLoc({funcOp->getLoc()}, spAttr));

  // We have done enough for the line table. Process variables only
  // if full debug info is required.
  if (debugLevel != llvm::codegenoptions::FullDebugInfo)
    return;

  uint32_t argNo = 1;
  funcOp.walk([&](fir::DeclareOp declOp) {
    handleDeclareOp(declOp, funcOp, fileAttr, spAttr, argNo);
  });
}

void AddDebugFoundationPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *context = &getContext();
  std::string inputFilePath("-");
  TypeConverter tyConverter(module);
  if (auto fileLoc = module.getLoc().dyn_cast<mlir::FileLineColLoc>())
    inputFilePath = fileLoc.getFilename().getValue();

  mlir::StringAttr producer = mlir::StringAttr::get(context, "Flang");
  mlir::LLVM::DIFileAttr fileAttr = mlir::LLVM::DIFileAttr::get(
      context, llvm::sys::path::filename(inputFilePath),
      llvm::sys::path::parent_path(inputFilePath));

  mlir::LLVM::DICompileUnitAttr cuAttr = mlir::LLVM::DICompileUnitAttr::get(
      context, mlir::DistinctAttr::create(mlir::UnitAttr::get(context)),
      llvm::dwarf::getLanguage("DW_LANG_Fortran95"), fileAttr, producer,
      /*isOptimized=*/false, getEmissionKind(debugLevel));

  module.walk([&](mlir::func::FuncOp funcOp) {
    handleFunctionOp(funcOp, fileAttr, cuAttr, inputFilePath);
  });

  // Process GlobalOp only if full debug info is required.
  if (debugLevel != llvm::codegenoptions::FullDebugInfo)
    return;

  module.walk([&](fir::GlobalOp globalOp) {
    handleGlobalOp(globalOp, fileAttr, cuAttr);
  });
}

std::unique_ptr<mlir::Pass>
fir::createAddDebugFoundationPass(unsigned debugLevel) {
  return std::make_unique<AddDebugFoundationPass>(debugLevel);
}

std::unique_ptr<mlir::Pass> fir::createAddDebugFoundationPass() {
  return std::make_unique<AddDebugFoundationPass>();
}
