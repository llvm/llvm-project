//===- CIRGenModule.cpp - Per-Module state for CIR generation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-translation-unit state used for CIR translation.
//
//===----------------------------------------------------------------------===//
#include "CIRGenCUDARuntime.h"
#include "CIRGenCXXABI.h"
#include "CIRGenCstEmitter.h"
#include "CIRGenFunction.h"
#include "CIRGenOpenMPRuntime.h"
#include "CIRGenTBAA.h"
#include "CIRGenTypes.h"
#include "CIRGenValue.h"
#include "TargetInfo.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/Cuda.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Basic/NoSanitizeList.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/CIR/CIRGenerator.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <numeric>

using namespace cir;
using namespace clang;
using namespace clang::CIRGen;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::SmallVector;
using llvm::StringRef;

static CIRGenCXXABI *createCXXABI(CIRGenModule &CGM) {
  switch (CGM.getASTContext().getCXXABIKind()) {
  case TargetCXXABI::GenericItanium:
  case TargetCXXABI::GenericAArch64:
  case TargetCXXABI::AppleARM64:
    return CreateCIRGenItaniumCXXABI(CGM);
  default:
    llvm_unreachable("invalid C++ ABI kind");
  }
}

CIRGenModule::CIRGenModule(mlir::MLIRContext &mlirContext,
                           clang::ASTContext &astContext,
                           const clang::CodeGenOptions &CGO,
                           DiagnosticsEngine &Diags)
    : builder(mlirContext, *this), astContext(astContext),
      langOpts(astContext.getLangOpts()), codeGenOpts(CGO),
      theModule{mlir::ModuleOp::create(builder.getUnknownLoc())}, Diags(Diags),
      target(astContext.getTargetInfo()), ABI(createCXXABI(*this)),
      genTypes{*this}, VTables{*this},
      openMPRuntime(new CIRGenOpenMPRuntime(*this)),
      cudaRuntime(new CIRGenCUDARuntime(*this)) {

  // Initialize CIR signed integer types cache.
  SInt8Ty = cir::IntType::get(&getMLIRContext(), 8, /*isSigned=*/true);
  SInt16Ty = cir::IntType::get(&getMLIRContext(), 16, /*isSigned=*/true);
  SInt32Ty = cir::IntType::get(&getMLIRContext(), 32, /*isSigned=*/true);
  SInt64Ty = cir::IntType::get(&getMLIRContext(), 64, /*isSigned=*/true);
  SInt128Ty = cir::IntType::get(&getMLIRContext(), 128, /*isSigned=*/true);

  // Initialize CIR unsigned integer types cache.
  UInt8Ty = cir::IntType::get(&getMLIRContext(), 8, /*isSigned=*/false);
  UInt16Ty = cir::IntType::get(&getMLIRContext(), 16, /*isSigned=*/false);
  UInt32Ty = cir::IntType::get(&getMLIRContext(), 32, /*isSigned=*/false);
  UInt64Ty = cir::IntType::get(&getMLIRContext(), 64, /*isSigned=*/false);
  UInt128Ty = cir::IntType::get(&getMLIRContext(), 128, /*isSigned=*/false);

  VoidTy = cir::VoidType::get(&getMLIRContext());

  // Initialize CIR pointer types cache.
  VoidPtrTy = cir::PointerType::get(&getMLIRContext(), VoidTy);
  VoidPtrPtrTy = cir::PointerType::get(&getMLIRContext(), VoidPtrTy);

  FP16Ty = cir::FP16Type::get(&getMLIRContext());
  BFloat16Ty = cir::BF16Type::get(&getMLIRContext());
  FloatTy = cir::SingleType::get(&getMLIRContext());
  DoubleTy = cir::DoubleType::get(&getMLIRContext());
  FP80Ty = cir::FP80Type::get(&getMLIRContext());
  FP128Ty = cir::FP128Type::get(&getMLIRContext());

  // TODO: PointerWidthInBits
  PointerAlignInBytes =
      astContext
          .toCharUnitsFromBits(
              astContext.getTargetInfo().getPointerAlign(LangAS::Default))
          .getQuantity();
  SizeSizeInBytes =
      astContext
          .toCharUnitsFromBits(astContext.getTargetInfo().getMaxPointerWidth())
          .getQuantity();
  // TODO: IntAlignInBytes
  UCharTy = cir::IntType::get(&getMLIRContext(),
                              astContext.getTargetInfo().getCharWidth(),
                              /*isSigned=*/false);
  UIntTy = cir::IntType::get(&getMLIRContext(),
                             astContext.getTargetInfo().getIntWidth(),
                             /*isSigned=*/false);
  UIntPtrTy = cir::IntType::get(&getMLIRContext(),
                                astContext.getTargetInfo().getMaxPointerWidth(),
                                /*isSigned=*/false);
  UInt8PtrTy = builder.getPointerTo(UInt8Ty);
  UInt8PtrPtrTy = builder.getPointerTo(UInt8PtrTy);
  AllocaInt8PtrTy = UInt8PtrTy;
  AllocaVoidPtrTy = VoidPtrTy;
  // TODO: GlobalsInt8PtrTy
  // TODO: ConstGlobalsPtrTy
  CIRAllocaAddressSpace = getTargetCIRGenInfo().getCIRAllocaAddressSpace();

  PtrDiffTy = cir::IntType::get(&getMLIRContext(),
                                astContext.getTargetInfo().getMaxPointerWidth(),
                                /*isSigned=*/true);

  if (langOpts.OpenCL) {
    createOpenCLRuntime();
  }

  cir::sob::SignedOverflowBehavior sob;
  switch (langOpts.getSignedOverflowBehavior()) {
  case clang::LangOptions::SignedOverflowBehaviorTy::SOB_Defined:
    sob = sob::SignedOverflowBehavior::defined;
    break;
  case clang::LangOptions::SignedOverflowBehaviorTy::SOB_Undefined:
    sob = sob::SignedOverflowBehavior::undefined;
    break;
  case clang::LangOptions::SignedOverflowBehaviorTy::SOB_Trapping:
    sob = sob::SignedOverflowBehavior::trapping;
    break;
  }

  // FIXME(cir): Implement a custom CIR Module Op and attributes to leverage
  // MLIR features.
  theModule->setAttr(cir::CIRDialect::getSOBAttrName(),
                     cir::SignedOverflowBehaviorAttr::get(&mlirContext, sob));
  auto lang = SourceLanguageAttr::get(&mlirContext, getCIRSourceLanguage());
  theModule->setAttr(cir::CIRDialect::getLangAttrName(),
                     cir::LangAttr::get(&mlirContext, lang));
  theModule->setAttr(cir::CIRDialect::getTripleAttrName(),
                     builder.getStringAttr(getTriple().str()));
  if (CGO.OptimizationLevel > 0 || CGO.OptimizeSize > 0)
    theModule->setAttr(cir::CIRDialect::getOptInfoAttrName(),
                       cir::OptInfoAttr::get(&mlirContext,
                                             CGO.OptimizationLevel,
                                             CGO.OptimizeSize));
  // Set the module name to be the name of the main file. TranslationUnitDecl
  // often contains invalid source locations and isn't a reliable source for the
  // module location.
  auto MainFileID = astContext.getSourceManager().getMainFileID();
  const FileEntry &MainFile =
      *astContext.getSourceManager().getFileEntryForID(MainFileID);
  auto Path = MainFile.tryGetRealPathName();
  if (!Path.empty()) {
    theModule.setSymName(Path);
    theModule->setLoc(mlir::FileLineColLoc::get(&mlirContext, Path,
                                                /*line=*/0,
                                                /*col=*/0));
  }

  // Set CUDA GPU binary handle.
  if (langOpts.CUDA) {
    std::string cudaBinaryName = codeGenOpts.CudaGpuBinaryFileName;
    if (!cudaBinaryName.empty()) {
      theModule->setAttr(
          cir::CIRDialect::getCUDABinaryHandleAttrName(),
          cir::CUDABinaryHandleAttr::get(&mlirContext, cudaBinaryName));
    }
  }

  if (langOpts.Sanitize.has(SanitizerKind::Thread) ||
      (!codeGenOpts.RelaxedAliasing && codeGenOpts.OptimizationLevel > 0)) {
    tbaa.reset(new CIRGenTBAA(&mlirContext, astContext, genTypes, theModule,
                              codeGenOpts, langOpts));
  }
}

CIRGenModule::~CIRGenModule() {}

bool CIRGenModule::isTypeConstant(QualType Ty, bool ExcludeCtor,
                                  bool ExcludeDtor) {
  if (!Ty.isConstant(astContext) && !Ty->isReferenceType())
    return false;

  if (astContext.getLangOpts().CPlusPlus) {
    if (const CXXRecordDecl *Record =
            astContext.getBaseElementType(Ty)->getAsCXXRecordDecl())
      return ExcludeCtor && !Record->hasMutableFields() &&
             (Record->hasTrivialDestructor() || ExcludeDtor);
  }

  return true;
}

/// FIXME: this could likely be a common helper and not necessarily related
/// with codegen.
/// Return the best known alignment for an unknown pointer to a
/// particular class.
CharUnits CIRGenModule::getClassPointerAlignment(const CXXRecordDecl *RD) {
  if (!RD->hasDefinition())
    return CharUnits::One(); // Hopefully won't be used anywhere.

  auto &layout = astContext.getASTRecordLayout(RD);

  // If the class is final, then we know that the pointer points to an
  // object of that type and can use the full alignment.
  if (RD->isEffectivelyFinal())
    return layout.getAlignment();

  // Otherwise, we have to assume it could be a subclass.
  return layout.getNonVirtualAlignment();
}

/// FIXME: this could likely be a common helper and not necessarily related
/// with codegen.
CharUnits CIRGenModule::getNaturalPointeeTypeAlignment(
    QualType ty, LValueBaseInfo *baseInfo, TBAAAccessInfo *tbaaInfo) {
  return getNaturalTypeAlignment(ty->getPointeeType(), baseInfo, tbaaInfo,
                                 /* forPointeeType= */ true);
}

/// FIXME: this could likely be a common helper and not necessarily related
/// with codegen.
/// TODO: Add TBAAAccessInfo
CharUnits CIRGenModule::getNaturalTypeAlignment(QualType T,
                                                LValueBaseInfo *BaseInfo,
                                                TBAAAccessInfo *tbaaInfo,
                                                bool forPointeeType) {
  if (tbaaInfo) {
    *tbaaInfo = getTBAAAccessInfo(T);
  }
  // FIXME: This duplicates logic in ASTContext::getTypeAlignIfKnown. But
  // that doesn't return the information we need to compute BaseInfo.

  // Honor alignment typedef attributes even on incomplete types.
  // We also honor them straight for C++ class types, even as pointees;
  // there's an expressivity gap here.
  if (auto TT = T->getAs<TypedefType>()) {
    if (auto Align = TT->getDecl()->getMaxAlignment()) {
      if (BaseInfo)
        *BaseInfo = LValueBaseInfo(AlignmentSource::AttributedType);
      return astContext.toCharUnitsFromBits(Align);
    }
  }

  bool AlignForArray = T->isArrayType();

  // Analyze the base element type, so we don't get confused by incomplete
  // array types.
  T = astContext.getBaseElementType(T);

  if (T->isIncompleteType()) {
    // We could try to replicate the logic from
    // ASTContext::getTypeAlignIfKnown, but nothing uses the alignment if the
    // type is incomplete, so it's impossible to test. We could try to reuse
    // getTypeAlignIfKnown, but that doesn't return the information we need
    // to set BaseInfo.  So just ignore the possibility that the alignment is
    // greater than one.
    if (BaseInfo)
      *BaseInfo = LValueBaseInfo(AlignmentSource::Type);
    return CharUnits::One();
  }

  if (BaseInfo)
    *BaseInfo = LValueBaseInfo(AlignmentSource::Type);

  CharUnits Alignment;
  const CXXRecordDecl *RD;
  if (T.getQualifiers().hasUnaligned()) {
    Alignment = CharUnits::One();
  } else if (forPointeeType && !AlignForArray &&
             (RD = T->getAsCXXRecordDecl())) {
    // For C++ class pointees, we don't know whether we're pointing at a
    // base or a complete object, so we generally need to use the
    // non-virtual alignment.
    Alignment = getClassPointerAlignment(RD);
  } else {
    Alignment = astContext.getTypeAlignInChars(T);
  }

  // Cap to the global maximum type alignment unless the alignment
  // was somehow explicit on the type.
  if (unsigned MaxAlign = astContext.getLangOpts().MaxTypeAlign) {
    if (Alignment.getQuantity() > MaxAlign &&
        !astContext.isAlignmentRequired(T))
      Alignment = CharUnits::fromQuantity(MaxAlign);
  }
  return Alignment;
}

bool CIRGenModule::MustBeEmitted(const ValueDecl *Global) {
  // Never defer when EmitAllDecls is specified.
  assert(!langOpts.EmitAllDecls && "EmitAllDecls NYI");
  assert(!codeGenOpts.KeepStaticConsts && "KeepStaticConsts NYI");

  return getASTContext().DeclMustBeEmitted(Global);
}

bool CIRGenModule::MayBeEmittedEagerly(const ValueDecl *Global) {
  // In OpenMP 5.0 variables and function may be marked as
  // device_type(host/nohost) and we should not emit them eagerly unless we sure
  // that they must be emitted on the host/device. To be sure we need to have
  // seen a declare target with an explicit mentioning of the function, we know
  // we have if the level of the declare target attribute is -1. Note that we
  // check somewhere else if we should emit this at all.
  if (langOpts.OpenMP >= 50 && !langOpts.OpenMPSimd) {
    std::optional<OMPDeclareTargetDeclAttr *> ActiveAttr =
        OMPDeclareTargetDeclAttr::getActiveAttr(Global);
    if (!ActiveAttr || (*ActiveAttr)->getLevel() != (unsigned)-1)
      return false;
  }

  const auto *FD = dyn_cast<FunctionDecl>(Global);
  if (FD) {
    // Implicit template instantiations may change linkage if they are later
    // explicitly instantiated, so they should not be emitted eagerly.
    // TODO(cir): do we care?
    assert(FD->getTemplateSpecializationKind() != TSK_ImplicitInstantiation &&
           "not implemented");
    assert(!FD->isTemplated() && "Templates NYI");
  }
  const auto *VD = dyn_cast<VarDecl>(Global);
  if (VD)
    // A definition of an inline constexpr static data member may change
    // linkage later if it's redeclared outside the class.
    // TODO(cir): do we care?
    assert(astContext.getInlineVariableDefinitionKind(VD) !=
               ASTContext::InlineVariableDefinitionKind::WeakUnknown &&
           "not implemented");

  // If OpenMP is enabled and threadprivates must be generated like TLS, delay
  // codegen for global variables, because they may be marked as threadprivate.
  if (langOpts.OpenMP && langOpts.OpenMPUseTLS &&
      getASTContext().getTargetInfo().isTLSSupported() &&
      isa<VarDecl>(Global) &&
      !Global->getType().isConstantStorage(getASTContext(), false, false) &&
      !OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(Global))
    return false;

  assert((FD || VD) &&
         "Only FunctionDecl and VarDecl should hit this path so far.");
  return true;
}

static bool shouldAssumeDSOLocal(const CIRGenModule &CGM,
                                 CIRGlobalValueInterface GV) {
  if (GV.hasLocalLinkage())
    return true;

  if (!GV.hasDefaultVisibility() && !GV.hasExternalWeakLinkage()) {
    return true;
  }

  // DLLImport explicitly marks the GV as external.
  // so it shouldn't be dso_local
  // But we don't have the info set now
  assert(!cir::MissingFeatures::setDLLImportDLLExport());

  const llvm::Triple &TT = CGM.getTriple();
  const auto &CGOpts = CGM.getCodeGenOpts();
  if (TT.isWindowsGNUEnvironment()) {
    // In MinGW, variables without DLLImport can still be automatically
    // imported from a DLL by the linker; don't mark variables that
    // potentially could come from another DLL as DSO local.

    // With EmulatedTLS, TLS variables can be autoimported from other DLLs
    // (and this actually happens in the public interface of libstdc++), so
    // such variables can't be marked as DSO local. (Native TLS variables
    // can't be dllimported at all, though.)
    llvm_unreachable("MinGW not supported here");
  }

  // On COFF, don't mark 'extern_weak' symbols as DSO local. If these symbols
  // remain unresolved in the link, they can be resolved to zero, which is
  // outside the current DSO.
  if (TT.isOSBinFormatCOFF() && GV.hasExternalWeakLinkage())
    return false;

  // Every other GV is local on COFF.
  // Make an exception for windows OS in the triple: Some firmware builds use
  // *-win32-macho triples. This (accidentally?) produced windows relocations
  // without GOT tables in older clang versions; Keep this behaviour.
  // FIXME: even thread local variables?
  if (TT.isOSBinFormatCOFF() || (TT.isOSWindows() && TT.isOSBinFormatMachO()))
    return true;

  // Only handle COFF and ELF for now.
  if (!TT.isOSBinFormatELF())
    return false;

  llvm::Reloc::Model RM = CGOpts.RelocationModel;
  const auto &LOpts = CGM.getLangOpts();
  if (RM != llvm::Reloc::Static && !LOpts.PIE) {
    // On ELF, if -fno-semantic-interposition is specified and the target
    // supports local aliases, there will be neither CC1
    // -fsemantic-interposition nor -fhalf-no-semantic-interposition. Set
    // dso_local on the function if using a local alias is preferable (can avoid
    // PLT indirection).
    if (!(isa<cir::FuncOp>(GV) && GV.canBenefitFromLocalAlias())) {
      return false;
    }
    return !(CGM.getLangOpts().SemanticInterposition ||
             CGM.getLangOpts().HalfNoSemanticInterposition);
  }

  // A definition cannot be preempted from an executable.
  if (!GV.isDeclarationForLinker())
    return true;

  // Most PIC code sequences that assume that a symbol is local cannot produce a
  // 0 if it turns out the symbol is undefined. While this is ABI and relocation
  // depended, it seems worth it to handle it here.
  if (RM == llvm::Reloc::PIC_ && GV.hasExternalWeakLinkage())
    return false;

  // PowerPC64 prefers TOC indirection to avoid copy relocations.
  if (TT.isPPC64())
    return false;

  if (CGOpts.DirectAccessExternalData) {
    // If -fdirect-access-external-data (default for -fno-pic), set dso_local
    // for non-thread-local variables. If the symbol is not defined in the
    // executable, a copy relocation will be needed at link time. dso_local is
    // excluded for thread-local variables because they generally don't support
    // copy relocations.
    if (auto gv = dyn_cast<cir::GlobalOp>(GV.getOperation()))
      if (!gv.getTlsModelAttr())
        return true;

    // -fno-pic sets dso_local on a function declaration to allow direct
    // accesses when taking its address (similar to a data symbol). If the
    // function is not defined in the executable, a canonical PLT entry will be
    // needed at link time. -fno-direct-access-external-data can avoid the
    // canonical PLT entry. We don't generalize this condition to -fpie/-fpic as
    // it could just cause trouble without providing perceptible benefits.
    if (isa<cir::FuncOp>(GV) && !CGOpts.NoPLT && RM == llvm::Reloc::Static)
      return true;
  }

  // If we can use copy relocations we can assume it is local.

  // Otherwise don't assume it is local.

  return false;
}

void CIRGenModule::setDSOLocal(CIRGlobalValueInterface GV) const {
  GV.setDSOLocal(shouldAssumeDSOLocal(*this, GV));
}

const ABIInfo &CIRGenModule::getABIInfo() {
  return getTargetCIRGenInfo().getABIInfo();
}

bool CIRGenModule::shouldEmitCUDAGlobalVar(const VarDecl *global) const {
  assert(langOpts.CUDA && "Should not be called by non-CUDA languages");
  // We need to emit host-side 'shadows' for all global
  // device-side variables because the CUDA runtime needs their
  // size and host-side address in order to provide access to
  // their device-side incarnations.

  if (global->hasAttr<CUDAConstantAttr>() ||
      global->hasAttr<CUDASharedAttr>() ||
      global->getType()->isCUDADeviceBuiltinSurfaceType() ||
      global->getType()->isCUDADeviceBuiltinTextureType()) {
    llvm_unreachable("NYI");
  }

  return !langOpts.CUDAIsDevice || global->hasAttr<CUDADeviceAttr>();
}

void CIRGenModule::emitGlobal(GlobalDecl GD) {
  llvm::TimeTraceScope scope("build CIR Global", [&]() -> std::string {
    auto *ND = dyn_cast<NamedDecl>(GD.getDecl());
    if (!ND)
      // TODO: How to print decls which is not named decl?
      return "Unnamed decl";

    std::string Name;
    llvm::raw_string_ostream OS(Name);
    ND->getNameForDiagnostic(OS, getASTContext().getPrintingPolicy(),
                             /*Qualified=*/true);
    return Name;
  });

  const auto *Global = cast<ValueDecl>(GD.getDecl());

  assert(!Global->hasAttr<IFuncAttr>() && "NYI");
  assert(!Global->hasAttr<CPUDispatchAttr>() && "NYI");

  if (langOpts.CUDA || langOpts.HIP) {
    // clang uses the same flag when building HIP code
    if (langOpts.CUDAIsDevice) {
      // This will implicitly mark templates and their
      // specializations as __host__ __device__.
      if (langOpts.OffloadImplicitHostDeviceTemplates)
        llvm_unreachable("NYI");

      // This maps some parallel standard libraries implicitly
      // to GPU, even when they are not marked __device__.
      if (langOpts.HIPStdPar)
        llvm_unreachable("NYI");

      // Global functions reside on device, so it shouldn't be skipped.
      if (!Global->hasAttr<CUDAGlobalAttr>() &&
          !Global->hasAttr<CUDADeviceAttr>())
        return;
    } else {
      // We must skip __device__ functions when compiling for host.
      if (!Global->hasAttr<CUDAHostAttr>() &&
          Global->hasAttr<CUDADeviceAttr>()) {
        return;
      }
    }

    if (const auto *VD = dyn_cast<VarDecl>(Global)) {
      if (!shouldEmitCUDAGlobalVar(VD))
        return;
    }
  }

  if (langOpts.OpenMP) {
    // If this is OpenMP, check if it is legal to emit this global normally.
    if (openMPRuntime && openMPRuntime->emitTargetGlobal(GD)) {
      assert(!cir::MissingFeatures::openMPRuntime());
      return;
    }
    if (auto *DRD = dyn_cast<OMPDeclareReductionDecl>(Global)) {
      assert(!cir::MissingFeatures::openMP());
      return;
    }
    if (auto *DMD = dyn_cast<OMPDeclareMapperDecl>(Global)) {
      assert(!cir::MissingFeatures::openMP());
      return;
    }
  }

  // Ignore declarations, they will be emitted on their first use.
  if (const auto *FD = dyn_cast<FunctionDecl>(Global)) {
    // Update deferred annotations with the latest declaration if the function
    // was already used or defined.
    if (FD->hasAttr<AnnotateAttr>()) {
      StringRef MangledName = getMangledName(GD);
      if (getGlobalValue(MangledName))
        deferredAnnotations[MangledName] = FD;
    }
    // Forward declarations are emitted lazily on first use.
    if (!FD->doesThisDeclarationHaveABody()) {
      if (!FD->doesDeclarationForceExternallyVisibleDefinition())
        return;

      llvm::StringRef MangledName = getMangledName(GD);

      // Compute the function info and CIR type.
      const auto &FI = getTypes().arrangeGlobalDeclaration(GD);
      mlir::Type Ty = getTypes().GetFunctionType(FI);

      GetOrCreateCIRFunction(MangledName, Ty, GD, /*ForVTable=*/false,
                             /*DontDefer=*/false);
      return;
    }
  } else {
    const auto *VD = cast<VarDecl>(Global);
    assert(VD->isFileVarDecl() && "Cannot emit local var decl as global.");
    if (VD->isThisDeclarationADefinition() != VarDecl::Definition &&
        !astContext.isMSStaticDataMemberInlineDefinition(VD)) {
      if (langOpts.OpenMP) {
        // Emit declaration of the must-be-emitted declare target variable.
        if (std::optional<OMPDeclareTargetDeclAttr::MapTypeTy> Res =
                OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(VD)) {
          assert(0 && "OMPDeclareTargetDeclAttr NYI");
        }
      }
      // If this declaration may have caused an inline variable definition to
      // change linkage, make sure that it's emitted.
      if (astContext.getInlineVariableDefinitionKind(VD) ==
          ASTContext::InlineVariableDefinitionKind::Strong)
        getAddrOfGlobalVar(VD);
      return;
    }
  }

  // Defer code generation to first use when possible, e.g. if this is an inline
  // function. If the global mjust always be emitted, do it eagerly if possible
  // to benefit from cache locality.
  if (MustBeEmitted(Global) && MayBeEmittedEagerly(Global)) {
    // Emit the definition if it can't be deferred.
    emitGlobalDefinition(GD);
    return;
  }

  // If we're deferring emission of a C++ variable with an initializer, remember
  // the order in which it appeared on the file.
  if (getLangOpts().CPlusPlus && isa<VarDecl>(Global) &&
      cast<VarDecl>(Global)->hasInit()) {
    DelayedCXXInitPosition[Global] = CXXGlobalInits.size();
    CXXGlobalInits.push_back(nullptr);
  }

  llvm::StringRef MangledName = getMangledName(GD);
  if (getGlobalValue(MangledName) != nullptr) {
    // The value has already been used and should therefore be emitted.
    addDeferredDeclToEmit(GD);
  } else if (MustBeEmitted(Global)) {
    // The value must be emitted, but cannot be emitted eagerly.
    assert(!MayBeEmittedEagerly(Global));
    addDeferredDeclToEmit(GD);
  } else {
    // Otherwise, remember that we saw a deferred decl with this name. The first
    // use of the mangled name will cause it to move into DeferredDeclsToEmit.
    DeferredDecls[MangledName] = GD;
  }
}

void CIRGenModule::emitGlobalFunctionDefinition(GlobalDecl GD,
                                                mlir::Operation *Op) {
  auto const *D = cast<FunctionDecl>(GD.getDecl());

  // Compute the function info and CIR type.
  const CIRGenFunctionInfo &FI = getTypes().arrangeGlobalDeclaration(GD);
  auto Ty = getTypes().GetFunctionType(FI);

  // Get or create the prototype for the function.
  auto Fn = dyn_cast_if_present<cir::FuncOp>(Op);
  if (!Fn || Fn.getFunctionType() != Ty) {
    Fn = GetAddrOfFunction(GD, Ty, /*ForVTable=*/false,
                           /*DontDefer=*/true, ForDefinition);
  }

  // Already emitted.
  if (!Fn.isDeclaration())
    return;

  setFunctionLinkage(GD, Fn);
  setGVProperties(Fn, D);
  // TODO(cir): MaubeHandleStaticInExternC
  // TODO(cir): maybeSetTrivialComdat
  // TODO(cir): setLLVMFunctionFEnvAttributes

  CIRGenFunction CGF{*this, builder};
  CurCGF = &CGF;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    CGF.generateCode(GD, Fn, FI);
  }
  CurCGF = nullptr;

  setNonAliasAttributes(GD, Fn);
  setCIRFunctionAttributesForDefinition(D, Fn);

  if (const ConstructorAttr *CA = D->getAttr<ConstructorAttr>())
    AddGlobalCtor(Fn, CA->getPriority());
  if (const DestructorAttr *DA = D->getAttr<DestructorAttr>())
    AddGlobalDtor(Fn, DA->getPriority(), true);

  if (D->getAttr<AnnotateAttr>())
    deferredAnnotations[getMangledName(GD)] = cast<ValueDecl>(D);
}

/// Track functions to be called before main() runs.
void CIRGenModule::AddGlobalCtor(cir::FuncOp Ctor, int Priority) {
  // FIXME(cir): handle LexOrder and Associated data upon testcases.
  //
  // Traditional LLVM codegen directly adds the function to the list of global
  // ctors. In CIR we just add a global_ctor attribute to the function. The
  // global list is created in LoweringPrepare.
  //
  // FIXME(from traditional LLVM): Type coercion of void()* types.
  Ctor->setAttr(
      Ctor.getGlobalCtorAttrName(),
      cir::GlobalCtorAttr::get(&getMLIRContext(), Ctor.getName(), Priority));
}

/// Add a function to the list that will be called when the module is unloaded.
void CIRGenModule::AddGlobalDtor(cir::FuncOp Dtor, int Priority,
                                 bool IsDtorAttrFunc) {
  assert(IsDtorAttrFunc && "NYI");
  if (codeGenOpts.RegisterGlobalDtorsWithAtExit &&
      (!getASTContext().getTargetInfo().getTriple().isOSAIX() ||
       IsDtorAttrFunc)) {
    llvm_unreachable("NYI");
  }

  // FIXME(from traditional LLVM): Type coercion of void()* types.
  Dtor->setAttr(
      Dtor.getGlobalDtorAttrName(),
      cir::GlobalDtorAttr::get(&getMLIRContext(), Dtor.getName(), Priority));
}

mlir::Operation *CIRGenModule::getGlobalValue(StringRef Name) {
  auto global = mlir::SymbolTable::lookupSymbolIn(theModule, Name);
  if (!global)
    return {};
  return global;
}

mlir::Value CIRGenModule::getGlobalValue(const Decl *D) {
  assert(CurCGF);
  return CurCGF->symbolTable.lookup(D);
}

cir::GlobalOp CIRGenModule::createGlobalOp(CIRGenModule &cgm,
                                           mlir::Location loc, StringRef name,
                                           mlir::Type t, bool isConstant,
                                           cir::AddressSpaceAttr addrSpace,
                                           mlir::Operation *insertPoint,
                                           cir::GlobalLinkageKind linkage) {
  cir::GlobalOp g;
  auto &builder = cgm.getBuilder();
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Some global emissions are triggered while emitting a function, e.g.
    // void s() { const char *s = "yolo"; ... }
    //
    // Be sure to insert global before the current function
    auto *curCGF = cgm.getCurrCIRGenFun();
    if (curCGF)
      builder.setInsertionPoint(curCGF->CurFn);

    g = builder.create<cir::GlobalOp>(loc, name, t, isConstant, linkage,
                                      addrSpace);
    if (!curCGF) {
      if (insertPoint)
        cgm.getModule().insert(insertPoint, g);
      else
        cgm.getModule().push_back(g);
    }

    // Default to private until we can judge based on the initializer,
    // since MLIR doesn't allow public declarations.
    mlir::SymbolTable::setSymbolVisibility(
        g, mlir::SymbolTable::Visibility::Private);
  }
  return g;
}

void CIRGenModule::setCommonAttributes(GlobalDecl GD, mlir::Operation *GV) {
  const Decl *D = GD.getDecl();
  if (isa_and_nonnull<NamedDecl>(D))
    setGVProperties(GV, dyn_cast<NamedDecl>(D));
  else
    assert(!cir::MissingFeatures::setDefaultVisibility());

  if (D && D->hasAttr<UsedAttr>())
    assert(!cir::MissingFeatures::addUsedOrCompilerUsedGlobal());

  if (const auto *VD = dyn_cast_if_present<VarDecl>(D);
      VD &&
      ((codeGenOpts.KeepPersistentStorageVariables &&
        (VD->getStorageDuration() == SD_Static ||
         VD->getStorageDuration() == SD_Thread)) ||
       (codeGenOpts.KeepStaticConsts && VD->getStorageDuration() == SD_Static &&
        VD->getType().isConstQualified())))
    assert(!cir::MissingFeatures::addUsedOrCompilerUsedGlobal());
}

void CIRGenModule::setNonAliasAttributes(GlobalDecl GD, mlir::Operation *GO) {
  const Decl *D = GD.getDecl();
  setCommonAttributes(GD, GO);

  if (D) {
    auto GV = llvm::dyn_cast_or_null<cir::GlobalOp>(GO);
    if (GV) {
      if (D->hasAttr<RetainAttr>())
        assert(!cir::MissingFeatures::addUsedGlobal());
      if (auto *SA = D->getAttr<PragmaClangBSSSectionAttr>())
        assert(!cir::MissingFeatures::addSectionAttributes());
      if (auto *SA = D->getAttr<PragmaClangDataSectionAttr>())
        assert(!cir::MissingFeatures::addSectionAttributes());
      if (auto *SA = D->getAttr<PragmaClangRodataSectionAttr>())
        assert(!cir::MissingFeatures::addSectionAttributes());
      if (auto *SA = D->getAttr<PragmaClangRelroSectionAttr>())
        assert(!cir::MissingFeatures::addSectionAttributes());
    }
    auto F = llvm::dyn_cast_or_null<cir::FuncOp>(GO);
    if (F) {
      if (D->hasAttr<RetainAttr>())
        assert(!cir::MissingFeatures::addUsedGlobal());
      if (auto *SA = D->getAttr<PragmaClangTextSectionAttr>())
        if (!D->getAttr<SectionAttr>())
          assert(!cir::MissingFeatures::setSectionForFuncOp());

      assert(!cir::MissingFeatures::updateCPUAndFeaturesAttributes());
    }

    if (const auto *CSA = D->getAttr<CodeSegAttr>()) {
      assert(!cir::MissingFeatures::setSectionForFuncOp());
      if (GV)
        GV.setSection(CSA->getName());
      if (F)
        assert(!cir::MissingFeatures::setSectionForFuncOp());
    } else if (const auto *SA = D->getAttr<SectionAttr>())
      if (GV)
        GV.setSection(SA->getName());
    if (F)
      assert(!cir::MissingFeatures::setSectionForFuncOp());
  }
  assert(!cir::MissingFeatures::setTargetAttributes());
}

static llvm::SmallVector<int64_t> indexesOfArrayAttr(mlir::ArrayAttr indexes) {
  llvm::SmallVector<int64_t> inds;

  for (mlir::Attribute i : indexes) {
    auto ind = dyn_cast<mlir::IntegerAttr>(i);
    assert(ind && "expect MLIR integer attribute");
    inds.push_back(ind.getValue().getSExtValue());
  }

  return inds;
}

static bool isViewOnGlobal(GlobalOp glob, GlobalViewAttr view) {
  return view.getSymbol().getValue() == glob.getSymName();
}

static GlobalViewAttr createNewGlobalView(CIRGenModule &CGM, GlobalOp newGlob,
                                          GlobalViewAttr attr,
                                          mlir::Type oldTy) {
  if (!attr.getIndices() || !isViewOnGlobal(newGlob, attr))
    return attr;

  llvm::SmallVector<int64_t> oldInds = indexesOfArrayAttr(attr.getIndices());
  llvm::SmallVector<int64_t> newInds;
  CIRGenBuilderTy &bld = CGM.getBuilder();
  const CIRDataLayout &layout = CGM.getDataLayout();
  mlir::MLIRContext *ctxt = bld.getContext();
  auto newTy = newGlob.getSymType();

  auto offset = bld.computeOffsetFromGlobalViewIndices(layout, oldTy, oldInds);
  bld.computeGlobalViewIndicesFromFlatOffset(offset, newTy, layout, newInds);
  cir::PointerType newPtrTy;

  if (isa<cir::StructType>(oldTy))
    newPtrTy = cir::PointerType::get(ctxt, newTy);
  else if (cir::ArrayType oldArTy = dyn_cast<cir::ArrayType>(oldTy))
    newPtrTy = dyn_cast<cir::PointerType>(attr.getType());

  if (newPtrTy)
    return bld.getGlobalViewAttr(newPtrTy, newGlob, newInds);

  llvm_unreachable("NYI");
}

static mlir::Attribute getNewInitValue(CIRGenModule &CGM, GlobalOp newGlob,
                                       mlir::Type oldTy, GlobalOp user,
                                       mlir::Attribute oldInit) {
  if (auto oldView = mlir::dyn_cast<cir::GlobalViewAttr>(oldInit)) {
    return createNewGlobalView(CGM, newGlob, oldView, oldTy);
  } else if (auto oldArray = mlir::dyn_cast<ConstArrayAttr>(oldInit)) {
    llvm::SmallVector<mlir::Attribute> newArray;
    auto eltsAttr = dyn_cast<mlir::ArrayAttr>(oldArray.getElts());
    for (auto elt : eltsAttr) {
      if (auto view = dyn_cast<GlobalViewAttr>(elt))
        newArray.push_back(createNewGlobalView(CGM, newGlob, view, oldTy));
      else if (auto view = dyn_cast<ConstArrayAttr>(elt))
        newArray.push_back(getNewInitValue(CGM, newGlob, oldTy, user, elt));
      else
        newArray.push_back(elt);
    }

    auto &builder = CGM.getBuilder();
    mlir::Attribute ar = mlir::ArrayAttr::get(builder.getContext(), newArray);
    return builder.getConstArray(ar, cast<cir::ArrayType>(oldArray.getType()));
  } else {
    llvm_unreachable("NYI");
  }
}

void CIRGenModule::replaceGlobal(cir::GlobalOp Old, cir::GlobalOp New) {
  assert(Old.getSymName() == New.getSymName() && "symbol names must match");
  // If the types does not match, update all references to Old to the new type.
  auto OldTy = Old.getSymType();
  auto NewTy = New.getSymType();
  cir::AddressSpaceAttr oldAS = Old.getAddrSpaceAttr();
  cir::AddressSpaceAttr newAS = New.getAddrSpaceAttr();
  // TODO(cir): If the AS differs, we should also update all references.
  if (oldAS != newAS) {
    llvm_unreachable("NYI");
  }

  if (OldTy != NewTy) {
    auto OldSymUses = Old.getSymbolUses(theModule.getOperation());
    if (OldSymUses.has_value()) {
      for (auto Use : *OldSymUses) {
        auto *UserOp = Use.getUser();
        assert((isa<cir::GetGlobalOp>(UserOp) || isa<cir::GlobalOp>(UserOp)) &&
               "GlobalOp symbol user is neither a GetGlobalOp nor a GlobalOp");

        if (auto GGO = dyn_cast<cir::GetGlobalOp>(Use.getUser())) {
          auto UseOpResultValue = GGO.getAddr();
          UseOpResultValue.setType(
              cir::PointerType::get(&getMLIRContext(), NewTy));

          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointAfter(GGO);
          mlir::Type ptrTy = builder.getPointerTo(OldTy);
          mlir::Value cast =
              builder.createBitcast(GGO->getLoc(), UseOpResultValue, ptrTy);
          UseOpResultValue.replaceAllUsesExcept(cast, cast.getDefiningOp());
        } else if (auto glob = dyn_cast<cir::GlobalOp>(UserOp)) {
          if (auto init = glob.getInitialValue()) {
            auto nw = getNewInitValue(*this, New, OldTy, glob, init.value());
            glob.setInitialValueAttr(nw);
          }
        }
      }
    }
  }

  // Remove old global from the module.
  Old.erase();
}

cir::TLS_Model CIRGenModule::GetDefaultCIRTLSModel() const {
  switch (getCodeGenOpts().getDefaultTLSModel()) {
  case CodeGenOptions::GeneralDynamicTLSModel:
    return cir::TLS_Model::GeneralDynamic;
  case CodeGenOptions::LocalDynamicTLSModel:
    return cir::TLS_Model::LocalDynamic;
  case CodeGenOptions::InitialExecTLSModel:
    return cir::TLS_Model::InitialExec;
  case CodeGenOptions::LocalExecTLSModel:
    return cir::TLS_Model::LocalExec;
  }
  llvm_unreachable("Invalid TLS model!");
}

void CIRGenModule::setTLSMode(mlir::Operation *Op, const VarDecl &D) const {
  assert(D.getTLSKind() && "setting TLS mode on non-TLS var!");

  auto TLM = GetDefaultCIRTLSModel();

  // Override the TLS model if it is explicitly specified.
  if (const TLSModelAttr *Attr = D.getAttr<TLSModelAttr>()) {
    llvm_unreachable("NYI");
  }

  auto global = dyn_cast<cir::GlobalOp>(Op);
  assert(global && "NYI for other operations");
  global.setTlsModel(TLM);
}

/// If the specified mangled name is not in the module,
/// create and return an mlir GlobalOp with the specified type (TODO(cir):
/// address space).
///
/// TODO(cir):
/// 1. If there is something in the module with the specified name, return
/// it potentially bitcasted to the right type.
///
/// 2. If D is non-null, it specifies a decl that correspond to this.  This is
/// used to set the attributes on the global when it is first created.
///
/// 3. If IsForDefinition is true, it is guaranteed that an actual global with
/// type Ty will be returned, not conversion of a variable with the same
/// mangled name but some other type.
cir::GlobalOp
CIRGenModule::getOrCreateCIRGlobal(StringRef MangledName, mlir::Type Ty,
                                   LangAS langAS, const VarDecl *D,
                                   ForDefinition_t IsForDefinition) {
  // Lookup the entry, lazily creating it if necessary.
  cir::GlobalOp Entry;
  if (auto *V = getGlobalValue(MangledName)) {
    assert(isa<cir::GlobalOp>(V) && "only supports GlobalOp for now");
    Entry = dyn_cast_or_null<cir::GlobalOp>(V);
  }

  cir::AddressSpaceAttr cirAS = builder.getAddrSpaceAttr(langAS);
  if (Entry) {
    auto entryCIRAS = Entry.getAddrSpaceAttr();
    if (WeakRefReferences.erase(Entry)) {
      if (D && !D->hasAttr<WeakAttr>()) {
        auto LT = cir::GlobalLinkageKind::ExternalLinkage;
        Entry.setLinkageAttr(
            cir::GlobalLinkageKindAttr::get(&getMLIRContext(), LT));
        mlir::SymbolTable::setSymbolVisibility(Entry, getMLIRVisibility(Entry));
      }
    }

    // Handle dropped DLL attributes.
    if (D && !D->hasAttr<clang::DLLImportAttr>() &&
        !D->hasAttr<clang::DLLExportAttr>())
      assert(!cir::MissingFeatures::setDLLStorageClass() && "NYI");

    if (langOpts.OpenMP && !langOpts.OpenMPSimd && D)
      getOpenMPRuntime().registerTargetGlobalVariable(D, Entry);

    if (Entry.getSymType() == Ty && entryCIRAS == cirAS)
      return Entry;

    // If there are two attempts to define the same mangled name, issue an
    // error.
    //
    // TODO(cir): look at mlir::GlobalValue::isDeclaration for all aspects of
    // recognizing the global as a declaration, for now only check if
    // initializer is present.
    if (IsForDefinition && !Entry.isDeclaration()) {
      GlobalDecl OtherGD;
      const VarDecl *OtherD;

      // Check that D is not yet in DiagnosedConflictingDefinitions is required
      // to make sure that we issue an error only once.
      if (D && lookupRepresentativeDecl(MangledName, OtherGD) &&
          (D->getCanonicalDecl() != OtherGD.getCanonicalDecl().getDecl()) &&
          (OtherD = dyn_cast<VarDecl>(OtherGD.getDecl())) &&
          OtherD->hasInit() &&
          DiagnosedConflictingDefinitions.insert(D).second) {
        getDiags().Report(D->getLocation(), diag::err_duplicate_mangled_name)
            << MangledName;
        getDiags().Report(OtherGD.getDecl()->getLocation(),
                          diag::note_previous_definition);
      }
    }

    // TODO(cir): LLVM codegen makes sure the result is of the correct type
    // by issuing a address space cast.
    if (entryCIRAS != cirAS)
      llvm_unreachable("NYI");

    // (If global is requested for a definition, we always need to create a new
    // global, not just return a bitcast.)
    if (!IsForDefinition)
      return Entry;
  }

  auto declCIRAS = builder.getAddrSpaceAttr(getGlobalVarAddressSpace(D));
  // TODO(cir): do we need to strip pointer casts for Entry?

  auto loc = getLoc(D->getSourceRange());

  // mlir::SymbolTable::Visibility::Public is the default, no need to explicitly
  // mark it as such.
  auto GV = CIRGenModule::createGlobalOp(*this, loc, MangledName, Ty,
                                         /*isConstant=*/false,
                                         /*addrSpace=*/declCIRAS,
                                         /*insertPoint=*/Entry.getOperation());

  // If we already created a global with the same mangled name (but different
  // type) before, replace it with the new global.
  if (Entry) {
    replaceGlobal(Entry, GV);
  }

  // This is the first use or definition of a mangled name.  If there is a
  // deferred decl with this name, remember that we need to emit it at the end
  // of the file.
  auto DDI = DeferredDecls.find(MangledName);
  if (DDI != DeferredDecls.end()) {
    // Move the potentially referenced deferred decl to the DeferredDeclsToEmit
    // list, and remove it from DeferredDecls (since we don't need it anymore).
    addDeferredDeclToEmit(DDI->second);
    DeferredDecls.erase(DDI);
  }

  // Handle things which are present even on external declarations.
  if (D) {
    if (langOpts.OpenMP && !langOpts.OpenMPSimd && D)
      getOpenMPRuntime().registerTargetGlobalVariable(D, Entry);

    // FIXME: This code is overly simple and should be merged with other global
    // handling.
    GV.setAlignmentAttr(getSize(astContext.getDeclAlign(D)));
    GV.setConstant(isTypeConstant(D->getType(), false, false));
    // TODO(cir): setLinkageForGV(GV, D);

    if (D->getTLSKind()) {
      if (D->getTLSKind() == VarDecl::TLS_Dynamic)
        llvm_unreachable("NYI");
      setTLSMode(GV, *D);
    }

    setGVProperties(GV, D);

    // If required by the ABI, treat declarations of static data members with
    // inline initializers as definitions.
    if (astContext.isMSStaticDataMemberInlineDefinition(D)) {
      assert(0 && "not implemented");
    }

    // Emit section information for extern variables.
    if (D->hasExternalStorage()) {
      if (const SectionAttr *SA = D->getAttr<SectionAttr>())
        GV.setSectionAttr(builder.getStringAttr(SA->getName()));
    }

    GV.setGlobalVisibilityAttr(getGlobalVisibilityAttrFromDecl(D));

    // Handle XCore specific ABI requirements.
    if (getTriple().getArch() == llvm::Triple::xcore)
      assert(0 && "not implemented");

    // Check if we a have a const declaration with an initializer, we maybe
    // able to emit it as available_externally to expose it's value to the
    // optimizer.
    if (getLangOpts().CPlusPlus && GV.isPublic() &&
        D->getType().isConstQualified() && GV.isDeclaration() &&
        !D->hasDefinition() && D->hasInit() && !D->hasAttr<DLLImportAttr>()) {
      assert(0 && "not implemented");
    }
  }

  // TODO(cir): if this method is used to handle functions we must have
  // something closer to GlobalValue::isDeclaration instead of checking for
  // initializer.
  if (GV.isDeclaration()) {
    // TODO(cir): set target attributes

    // External HIP managed variables needed to be recorded for transformation
    // in both device and host compilations.
    // External HIP managed variables needed to be recorded for transformation
    // in both device and host compilations.
    if (getLangOpts().CUDA && D && D->hasAttr<HIPManagedAttr>() &&
        D->hasExternalStorage())
      llvm_unreachable("NYI");
  }

  // TODO(cir): address space cast when needed for DAddrSpace.
  return GV;
}

cir::GlobalOp
CIRGenModule::getOrCreateCIRGlobal(const VarDecl *D, mlir::Type Ty,
                                   ForDefinition_t IsForDefinition) {
  assert(D->hasGlobalStorage() && "Not a global variable");
  QualType ASTTy = D->getType();
  if (!Ty)
    Ty = getTypes().convertTypeForMem(ASTTy);

  StringRef MangledName = getMangledName(D);
  return getOrCreateCIRGlobal(MangledName, Ty, ASTTy.getAddressSpace(), D,
                              IsForDefinition);
}

/// Return the mlir::Value for the address of the given global variable. If Ty
/// is non-null and if the global doesn't exist, then it will be created with
/// the specified type instead of whatever the normal requested type would be.
/// If IsForDefinition is true, it is guaranteed that an actual global with type
/// Ty will be returned, not conversion of a variable with the same mangled name
/// but some other type.
mlir::Value CIRGenModule::getAddrOfGlobalVar(const VarDecl *D, mlir::Type Ty,
                                             ForDefinition_t IsForDefinition) {
  assert(D->hasGlobalStorage() && "Not a global variable");
  QualType ASTTy = D->getType();
  if (!Ty)
    Ty = getTypes().convertTypeForMem(ASTTy);

  bool tlsAccess = D->getTLSKind() != VarDecl::TLS_None;
  auto g = getOrCreateCIRGlobal(D, Ty, IsForDefinition);
  auto ptrTy = builder.getPointerTo(g.getSymType(), g.getAddrSpaceAttr());
  return builder.create<cir::GetGlobalOp>(getLoc(D->getSourceRange()), ptrTy,
                                          g.getSymName(), tlsAccess);
}

cir::GlobalViewAttr
CIRGenModule::getAddrOfGlobalVarAttr(const VarDecl *D, mlir::Type Ty,
                                     ForDefinition_t IsForDefinition) {
  assert(D->hasGlobalStorage() && "Not a global variable");
  QualType ASTTy = D->getType();
  if (!Ty)
    Ty = getTypes().convertTypeForMem(ASTTy);

  auto globalOp = getOrCreateCIRGlobal(D, Ty, IsForDefinition);
  auto ptrTy = builder.getPointerTo(globalOp.getSymType());
  return builder.getGlobalViewAttr(ptrTy, globalOp);
}

mlir::Operation *CIRGenModule::getWeakRefReference(const ValueDecl *VD) {
  const AliasAttr *AA = VD->getAttr<AliasAttr>();
  assert(AA && "No alias?");

  // See if there is already something with the target's name in the module.
  mlir::Operation *Entry = getGlobalValue(AA->getAliasee());
  if (Entry) {
    assert((isa<cir::GlobalOp>(Entry) || isa<cir::FuncOp>(Entry)) &&
           "weak ref should be against a global variable or function");
    return Entry;
  }

  mlir::Type DeclTy = getTypes().convertTypeForMem(VD->getType());
  if (mlir::isa<cir::FuncType>(DeclTy)) {
    auto F = GetOrCreateCIRFunction(AA->getAliasee(), DeclTy,
                                    GlobalDecl(cast<FunctionDecl>(VD)),
                                    /*ForVtable=*/false);
    F.setLinkage(cir::GlobalLinkageKind::ExternalWeakLinkage);
    WeakRefReferences.insert(F);
    return F;
  }

  llvm_unreachable("GlobalOp NYI");
}

/// TODO(cir): looks like part of this code can be part of a common AST
/// helper betweem CIR and LLVM codegen.
template <typename SomeDecl>
void CIRGenModule::maybeHandleStaticInExternC(const SomeDecl *D,
                                              cir::GlobalOp GV) {
  if (!getLangOpts().CPlusPlus)
    return;

  // Must have 'used' attribute, or else inline assembly can't rely on
  // the name existing.
  if (!D->template hasAttr<UsedAttr>())
    return;

  // Must have internal linkage and an ordinary name.
  if (!D->getIdentifier() || D->getFormalLinkage() != Linkage::Internal)
    return;

  // Must be in an extern "C" context. Entities declared directly within
  // a record are not extern "C" even if the record is in such a context.
  const SomeDecl *First = D->getFirstDecl();
  if (First->getDeclContext()->isRecord() || !First->isInExternCContext())
    return;

  // TODO(cir):
  // OK, this is an internal linkage entity inside an extern "C" linkage
  // specification. Make a note of that so we can give it the "expected"
  // mangled name if nothing else is using that name.
  //
  // If we have multiple internal linkage entities with the same name
  // in extern "C" regions, none of them gets that name.
  assert(0 && "not implemented");
}

void CIRGenModule::emitGlobalVarDefinition(const clang::VarDecl *D,
                                           bool IsTentative) {
  // TODO(cir):
  // OpenCL global variables of sampler type are translated to function calls,
  // therefore no need to be translated.
  // If this is OpenMP device, check if it is legal to emit this global
  // normally.
  QualType ASTTy = D->getType();
  if ((getLangOpts().OpenCL && ASTTy->isSamplerT()) ||
      getLangOpts().OpenMPIsTargetDevice)
    llvm_unreachable("not implemented");

  // TODO(cir): LLVM's codegen uses a llvm::TrackingVH here. Is that
  // necessary here for CIR gen?
  mlir::Attribute Init;
  bool NeedsGlobalCtor = false;
  // Whether the definition of the variable is available externally.
  // If yes, we shouldn't emit the GloablCtor and GlobalDtor for the variable
  // since this is the job for its original source.
  bool IsDefinitionAvailableExternally =
      astContext.GetGVALinkageForVariable(D) == GVA_AvailableExternally;
  bool NeedsGlobalDtor =
      !IsDefinitionAvailableExternally &&
      D->needsDestruction(astContext) == QualType::DK_cxx_destructor;

  // It is helpless to emit the definition for an available_externally variable
  // which can't be marked as const.
  // We don't need to check if it needs global ctor or dtor. See the above
  // comment for ideas.
  if (IsDefinitionAvailableExternally &&
      (!D->hasConstantInitialization() ||
       // TODO: Update this when we have interface to check constexpr
       // destructor.
       D->needsDestruction(getASTContext()) ||
       !D->getType().isConstantStorage(getASTContext(), true, true)))
    return;

  const VarDecl *InitDecl;
  const Expr *InitExpr = D->getAnyInitializer(InitDecl);

  std::optional<ConstantEmitter> emitter;

  // CUDA E.2.4.1 "__shared__ variables cannot have an initialization
  // as part of their declaration."  Sema has already checked for
  // error cases, so we just need to set Init to UndefValue.
  bool IsCUDASharedVar =
      getLangOpts().CUDAIsDevice && D->hasAttr<CUDASharedAttr>();
  // Shadows of initialized device-side global variables are also left
  // undefined.
  // Managed Variables should be initialized on both host side and device side.
  bool IsCUDAShadowVar =
      !getLangOpts().CUDAIsDevice && !D->hasAttr<HIPManagedAttr>() &&
      (D->hasAttr<CUDAConstantAttr>() || D->hasAttr<CUDADeviceAttr>() ||
       D->hasAttr<CUDASharedAttr>());
  bool IsCUDADeviceShadowVar =
      getLangOpts().CUDAIsDevice && !D->hasAttr<HIPManagedAttr>() &&
      (D->getType()->isCUDADeviceBuiltinSurfaceType() ||
       D->getType()->isCUDADeviceBuiltinTextureType());
  if (getLangOpts().CUDA &&
      (IsCUDASharedVar || IsCUDAShadowVar || IsCUDADeviceShadowVar))
    assert(0 && "not implemented");
  else if (D->hasAttr<LoaderUninitializedAttr>())
    assert(0 && "not implemented");
  else if (!InitExpr) {
    // This is a tentative definition; tentative definitions are
    // implicitly initialized with { 0 }.
    //
    // Note that tentative definitions are only emitted at the end of
    // a translation unit, so they should never have incomplete
    // type. In addition, EmitTentativeDefinition makes sure that we
    // never attempt to emit a tentative definition if a real one
    // exists. A use may still exists, however, so we still may need
    // to do a RAUW.
    assert(!ASTTy->isIncompleteType() && "Unexpected incomplete type");
    Init = builder.getZeroInitAttr(convertType(D->getType()));
  } else {
    initializedGlobalDecl = GlobalDecl(D);
    emitter.emplace(*this);
    auto Initializer = emitter->tryEmitForInitializer(*InitDecl);
    if (!Initializer) {
      QualType T = InitExpr->getType();
      if (D->getType()->isReferenceType())
        T = D->getType();

      if (getLangOpts().CPlusPlus) {
        if (InitDecl->hasFlexibleArrayInit(astContext))
          ErrorUnsupported(D, "flexible array initializer");
        Init = builder.getZeroInitAttr(convertType(T));
        if (!IsDefinitionAvailableExternally)
          NeedsGlobalCtor = true;
      } else {
        ErrorUnsupported(D, "static initializer");
      }
    } else {
      Init = Initializer;
      // We don't need an initializer, so remove the entry for the delayed
      // initializer position (just in case this entry was delayed) if we
      // also don't need to register a destructor.
      if (getLangOpts().CPlusPlus && !NeedsGlobalDtor)
        DelayedCXXInitPosition.erase(D);
    }
  }

  mlir::Type InitType;
  // If the initializer attribute is a SymbolRefAttr it means we are
  // initializing the global based on a global constant.
  //
  // TODO(cir): create another attribute to contain the final type and abstract
  // away SymbolRefAttr.
  if (auto symAttr = mlir::dyn_cast<mlir::SymbolRefAttr>(Init)) {
    auto cstGlobal = mlir::SymbolTable::lookupSymbolIn(theModule, symAttr);
    assert(isa<cir::GlobalOp>(cstGlobal) &&
           "unaware of other symbol providers");
    auto g = cast<cir::GlobalOp>(cstGlobal);
    auto arrayTy = mlir::dyn_cast<cir::ArrayType>(g.getSymType());
    // TODO(cir): pointer to array decay. Should this be modeled explicitly in
    // CIR?
    if (arrayTy)
      InitType = cir::PointerType::get(&getMLIRContext(), arrayTy.getEltType());
  } else {
    assert(mlir::isa<mlir::TypedAttr>(Init) && "This should have a type");
    auto TypedInitAttr = mlir::cast<mlir::TypedAttr>(Init);
    InitType = TypedInitAttr.getType();
  }
  assert(!mlir::isa<mlir::NoneType>(InitType) && "Should have a type by now");

  auto Entry = getOrCreateCIRGlobal(D, InitType, ForDefinition_t(!IsTentative));
  // TODO(cir): Strip off pointer casts from Entry if we get them?

  // TODO(cir): use GlobalValue interface
  assert(dyn_cast<GlobalOp>(&Entry) && "FuncOp not supported here");
  auto GV = Entry;

  // We have a definition after a declaration with the wrong type.
  // We must make a new GlobalVariable* and update everything that used OldGV
  // (a declaration or tentative definition) with the new GlobalVariable*
  // (which will be a definition).
  //
  // This happens if there is a prototype for a global (e.g.
  // "extern int x[];") and then a definition of a different type (e.g.
  // "int x[10];"). This also happens when an initializer has a different type
  // from the type of the global (this happens with unions).
  if (!GV || GV.getSymType() != InitType) {
    // TODO(cir): this should include an address space check as well.
    assert(0 && "not implemented");
  }

  maybeHandleStaticInExternC(D, GV);

  if (D->hasAttr<AnnotateAttr>())
    addGlobalAnnotations(D, GV);

  // Set CIR's linkage type as appropriate.
  cir::GlobalLinkageKind Linkage =
      getCIRLinkageVarDefinition(D, /*IsConstant=*/false);

  // TODO(cir):
  // CUDA B.2.1 "The __device__ qualifier declares a variable that resides on
  // the device. [...]"
  // CUDA B.2.2 "The __constant__ qualifier, optionally used together with
  // __device__, declares a variable that: [...]

  // Set initializer and finalize emission
  CIRGenModule::setInitializer(GV, Init);
  if (emitter)
    emitter->finalize(GV);

  // TODO(cir): If it is safe to mark the global 'constant', do so now.
  GV.setConstant(!NeedsGlobalCtor && !NeedsGlobalDtor &&
                 isTypeConstant(D->getType(), true, true));

  // If it is in a read-only section, mark it 'constant'.
  if (const SectionAttr *SA = D->getAttr<SectionAttr>())
    GV.setSectionAttr(builder.getStringAttr(SA->getName()));

  GV.setGlobalVisibilityAttr(getGlobalVisibilityAttrFromDecl(D));

  // TODO(cir):
  // GV->setAlignment(getContext().getDeclAlign(D).getAsAlign());

  // On Darwin, unlike other Itanium C++ ABI platforms, the thread-wrapper
  // function is only defined alongside the variable, not also alongside
  // callers. Normally, all accesses to a thread_local go through the
  // thread-wrapper in order to ensure initialization has occurred, underlying
  // variable will never be used other than the thread-wrapper, so it can be
  // converted to internal linkage.
  //
  // However, if the variable has the 'constinit' attribute, it _can_ be
  // referenced directly, without calling the thread-wrapper, so the linkage
  // must not be changed.
  //
  // Additionally, if the variable isn't plain external linkage, e.g. if it's
  // weak or linkonce, the de-duplication semantics are important to preserve,
  // so we don't change the linkage.
  if (D->getTLSKind() == VarDecl::TLS_Dynamic && GV.isPublic() &&
      astContext.getTargetInfo().getTriple().isOSDarwin() &&
      !D->hasAttr<ConstInitAttr>()) {
    // TODO(cir): set to mlir::SymbolTable::Visibility::Private once we have
    // testcases.
    assert(0 && "not implemented");
  }

  // Set CIR linkage and DLL storage class.
  GV.setLinkage(Linkage);
  // FIXME(cir): setLinkage should likely set MLIR's visibility automatically.
  GV.setVisibility(getMLIRVisibilityFromCIRLinkage(Linkage));
  // TODO(cir): handle DLL storage classes in CIR?
  if (D->hasAttr<DLLImportAttr>())
    assert(!cir::MissingFeatures::setDLLStorageClass());
  else if (D->hasAttr<DLLExportAttr>())
    assert(!cir::MissingFeatures::setDLLStorageClass());
  else
    assert(!cir::MissingFeatures::setDLLStorageClass());

  if (Linkage == cir::GlobalLinkageKind::CommonLinkage) {
    // common vars aren't constant even if declared const.
    GV.setConstant(false);
    // Tentative definition of global variables may be initialized with
    // non-zero null pointers. In this case they should have weak linkage
    // since common linkage must have zero initializer and must not have
    // explicit section therefore cannot have non-zero initial value.
    auto Initializer = GV.getInitialValue();
    if (Initializer && !getBuilder().isNullValue(*Initializer))
      GV.setLinkage(cir::GlobalLinkageKind::WeakAnyLinkage);
  }

  setNonAliasAttributes(D, GV);

  if (D->getTLSKind() && !GV.getTlsModelAttr()) {
    if (D->getTLSKind() == VarDecl::TLS_Dynamic)
      llvm_unreachable("NYI");
    setTLSMode(GV, *D);
  }

  maybeSetTrivialComdat(*D, GV);

  // TODO(cir):
  // Emit the initializer function if necessary.
  if (NeedsGlobalCtor || NeedsGlobalDtor) {
    globalOpContext = GV;
    emitCXXGlobalVarDeclInitFunc(D, GV, NeedsGlobalCtor);
    globalOpContext = nullptr;
  }

  // TODO(cir): sanitizers (reportGlobalToASan) and global variable debug
  // information.
  assert(!cir::MissingFeatures::sanitizeOther());
  assert(!cir::MissingFeatures::generateDebugInfo());
}

void CIRGenModule::emitGlobalDefinition(GlobalDecl GD, mlir::Operation *Op) {
  const auto *D = cast<ValueDecl>(GD.getDecl());
  if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
    // At -O0, don't generate CIR for functions with available_externally
    // linkage.
    if (!shouldEmitFunction(GD))
      return;

    if (const auto *Method = dyn_cast<CXXMethodDecl>(D)) {
      // Make sure to emit the definition(s) before we emit the thunks. This is
      // necessary for the generation of certain thunks.
      if (isa<CXXConstructorDecl>(Method) || isa<CXXDestructorDecl>(Method))
        ABI->emitCXXStructor(GD);
      else if (FD->isMultiVersion())
        llvm_unreachable("NYI");
      else
        emitGlobalFunctionDefinition(GD, Op);

      if (Method->isVirtual())
        getVTables().emitThunks(GD);

      return;
    }

    if (FD->isMultiVersion())
      llvm_unreachable("NYI");
    emitGlobalFunctionDefinition(GD, Op);
    return;
  }

  if (const auto *VD = dyn_cast<VarDecl>(D)) {
    return emitGlobalVarDefinition(VD, !VD->hasDefinition());
  }

  llvm_unreachable("Invalid argument to emitGlobalDefinition()");
}

mlir::Attribute
CIRGenModule::getConstantArrayFromStringLiteral(const StringLiteral *E) {
  assert(!E->getType()->isPointerType() && "Strings are always arrays");

  // Don't emit it as the address of the string, emit the string data itself
  // as an inline array.
  if (E->getCharByteWidth() == 1) {
    SmallString<64> Str(E->getString());

    // Resize the string to the right size, which is indicated by its type.
    const ConstantArrayType *CAT =
        astContext.getAsConstantArrayType(E->getType());
    auto finalSize = CAT->getSize().getZExtValue();
    Str.resize(finalSize);

    auto eltTy = convertType(CAT->getElementType());
    return builder.getString(Str, eltTy, finalSize);
  }

  auto arrayTy = mlir::dyn_cast<cir::ArrayType>(convertType(E->getType()));
  assert(arrayTy && "string literals must be emitted as an array type");

  auto arrayEltTy = mlir::dyn_cast<cir::IntType>(arrayTy.getEltType());
  assert(arrayEltTy &&
         "string literal elements must be emitted as integral type");

  auto arraySize = arrayTy.getSize();
  auto literalSize = E->getLength();

  // Collect the code units.
  SmallVector<uint32_t, 32> elementValues;
  elementValues.reserve(arraySize);
  for (unsigned i = 0; i < literalSize; ++i)
    elementValues.push_back(E->getCodeUnit(i));
  elementValues.resize(arraySize);

  // If the string is full of null bytes, emit a #cir.zero instead.
  if (std::all_of(elementValues.begin(), elementValues.end(),
                  [](uint32_t x) { return x == 0; }))
    return builder.getZeroAttr(arrayTy);

  // Otherwise emit a constant array holding the characters.
  SmallVector<mlir::Attribute, 32> elements;
  elements.reserve(arraySize);
  for (uint64_t i = 0; i < arraySize; ++i)
    elements.push_back(cir::IntAttr::get(arrayEltTy, elementValues[i]));

  auto elementsAttr = mlir::ArrayAttr::get(&getMLIRContext(), elements);
  return builder.getConstArray(elementsAttr, arrayTy);
}

// TODO(cir): this could be a common AST helper for both CIR and LLVM codegen.
LangAS CIRGenModule::getGlobalConstantAddressSpace() const {
  // OpenCL v1.2 s6.5.3: a string literal is in the constant address space.
  if (getLangOpts().OpenCL)
    return LangAS::opencl_constant;
  if (getLangOpts().SYCLIsDevice)
    return LangAS::sycl_global;
  if (auto AS = getTarget().getConstantAddressSpace())
    return AS.value();
  return LangAS::Default;
}

// TODO(cir): this could be a common AST helper for both CIR and LLVM codegen.
LangAS CIRGenModule::getLangTempAllocaAddressSpace() const {
  if (getLangOpts().OpenCL)
    return LangAS::opencl_private;

  // For temporaries inside functions, CUDA treats them as normal variables.
  // LangAS::cuda_device, on the other hand, is reserved for those variables
  // explicitly marked with __device__.
  if (getLangOpts().CUDAIsDevice)
    return LangAS::Default;

  if (getLangOpts().SYCLIsDevice ||
      (getLangOpts().OpenMP && getLangOpts().OpenMPIsTargetDevice))
    llvm_unreachable("NYI");
  return LangAS::Default;
}

static cir::GlobalOp
generateStringLiteral(mlir::Location loc, mlir::TypedAttr C,
                      cir::GlobalLinkageKind LT, CIRGenModule &CGM,
                      StringRef GlobalName, CharUnits Alignment) {
  cir::AddressSpaceAttr addrSpaceAttr =
      CGM.getBuilder().getAddrSpaceAttr(CGM.getGlobalConstantAddressSpace());

  // Create a global variable for this string
  // FIXME(cir): check for insertion point in module level.
  auto GV = CIRGenModule::createGlobalOp(CGM, loc, GlobalName, C.getType(),
                                         !CGM.getLangOpts().WritableStrings,
                                         addrSpaceAttr);

  // Set up extra information and add to the module
  GV.setAlignmentAttr(CGM.getSize(Alignment));
  GV.setLinkageAttr(
      cir::GlobalLinkageKindAttr::get(CGM.getBuilder().getContext(), LT));
  CIRGenModule::setInitializer(GV, C);
  // TODO(cir)
  assert(!cir::MissingFeatures::threadLocal() && "NYI");
  assert(!cir::MissingFeatures::unnamedAddr() && "NYI");
  if (GV.isWeakForLinker()) {
    assert(CGM.supportsCOMDAT() && "Only COFF uses weak string literals");
    GV.setComdat(true);
  }
  CGM.setDSOLocal(static_cast<mlir::Operation *>(GV));
  return GV;
}

/// Return a pointer to a constant array for the given string literal.
cir::GlobalViewAttr
CIRGenModule::getAddrOfConstantStringFromLiteral(const StringLiteral *S,
                                                 StringRef Name) {
  CharUnits Alignment =
      astContext.getAlignOfGlobalVarInChars(S->getType(), /*VD=*/nullptr);

  mlir::Attribute C = getConstantArrayFromStringLiteral(S);

  cir::GlobalOp GV;
  if (!getLangOpts().WritableStrings && ConstantStringMap.count(C)) {
    GV = ConstantStringMap[C];
    // The bigger alignment always wins.
    if (!GV.getAlignment() ||
        uint64_t(Alignment.getQuantity()) > *GV.getAlignment())
      GV.setAlignmentAttr(getSize(Alignment));
  } else {
    SmallString<256> StringNameBuffer = Name;
    llvm::raw_svector_ostream Out(StringNameBuffer);
    if (StringLiteralCnt)
      Out << '.' << StringLiteralCnt;
    Name = Out.str();
    StringLiteralCnt++;

    SmallString<256> MangledNameBuffer;
    StringRef GlobalVariableName;
    auto LT = cir::GlobalLinkageKind::ExternalLinkage;

    // Mangle the string literal if that's how the ABI merges duplicate strings.
    // Don't do it if they are writable, since we don't want writes in one TU to
    // affect strings in another.
    if (getCXXABI().getMangleContext().shouldMangleStringLiteral(S) &&
        !getLangOpts().WritableStrings) {
      assert(0 && "not implemented");
    } else {
      LT = cir::GlobalLinkageKind::PrivateLinkage;
      GlobalVariableName = Name;
    }

    auto loc = getLoc(S->getSourceRange());
    auto typedC = llvm::dyn_cast<mlir::TypedAttr>(C);
    if (!typedC)
      llvm_unreachable("this should never be untyped at this point");
    GV = generateStringLiteral(loc, typedC, LT, *this, GlobalVariableName,
                               Alignment);
    setDSOLocal(static_cast<mlir::Operation *>(GV));
    ConstantStringMap[C] = GV;

    assert(!cir::MissingFeatures::reportGlobalToASan() && "NYI");
  }

  auto ArrayTy = mlir::dyn_cast<cir::ArrayType>(GV.getSymType());
  assert(ArrayTy && "String literal must be array");
  auto PtrTy =
      getBuilder().getPointerTo(ArrayTy.getEltType(), GV.getAddrSpaceAttr());

  return builder.getGlobalViewAttr(PtrTy, GV);
}

void CIRGenModule::emitDeclContext(const DeclContext *DC) {
  for (auto *I : DC->decls()) {
    // Unlike other DeclContexts, the contents of an ObjCImplDecl at TU scope
    // are themselves considered "top-level", so EmitTopLevelDecl on an
    // ObjCImplDecl does not recursively visit them. We need to do that in
    // case they're nested inside another construct (LinkageSpecDecl /
    // ExportDecl) that does stop them from being considered "top-level".
    if (auto *OID = dyn_cast<ObjCImplDecl>(I))
      llvm_unreachable("NYI");

    emitTopLevelDecl(I);
  }
}

void CIRGenModule::emitLinkageSpec(const LinkageSpecDecl *LSD) {
  if (LSD->getLanguage() != LinkageSpecLanguageIDs::C &&
      LSD->getLanguage() != LinkageSpecLanguageIDs::CXX) {
    llvm_unreachable("unsupported linkage spec");
    return;
  }
  emitDeclContext(LSD);
}

mlir::Operation *
CIRGenModule::getAddrOfGlobalTemporary(const MaterializeTemporaryExpr *expr,
                                       const Expr *init) {
  assert((expr->getStorageDuration() == SD_Static ||
          expr->getStorageDuration() == SD_Thread) &&
         "not a global temporary");
  const auto *varDecl = cast<VarDecl>(expr->getExtendingDecl());

  // If we're not materializing a subobject of the temporay, keep the
  // cv-qualifiers from the type of the MaterializeTemporaryExpr.
  QualType materializedType = init->getType();
  if (init == expr->getSubExpr())
    materializedType = expr->getType();

  [[maybe_unused]] CharUnits align =
      getASTContext().getTypeAlignInChars(materializedType);

  auto insertResult = materializedGlobalTemporaryMap.insert({expr, nullptr});
  if (!insertResult.second) {
    llvm_unreachable("NYI");
  }

  // FIXME: If an externally-visible declaration extends multiple temporaries,
  // we need to give each temporary the same name in every translation unit (and
  // we also need to make the temporaries externally-visible).
  llvm::SmallString<256> name;
  llvm::raw_svector_ostream out(name);
  getCXXABI().getMangleContext().mangleReferenceTemporary(
      varDecl, expr->getManglingNumber(), out);

  APValue *value = nullptr;
  if (expr->getStorageDuration() == SD_Static && varDecl->evaluateValue()) {
    // If the initializer of the extending declaration is a constant
    // initializer, we should have a cached constant initializer for this
    // temporay. Note taht this m ight have a different value from the value
    // computed by evaluating the initializer if the surrounding constant
    // expression modifies the temporary.
    value = expr->getOrCreateValue(false);
  }

  // Try evaluating it now, it might have a constant initializer
  Expr::EvalResult evalResult;
  if (!value && init->EvaluateAsRValue(evalResult, getASTContext()) &&
      !evalResult.hasSideEffects())
    value = &evalResult.Val;

  LangAS addrSpace = getGlobalVarAddressSpace(varDecl);

  std::optional<ConstantEmitter> emitter;
  mlir::Attribute initialValue = nullptr;
  bool isConstant = false;
  mlir::Type type;
  if (value) {
    emitter.emplace(*this);
    initialValue =
        emitter->emitForInitializer(*value, addrSpace, materializedType);

    isConstant = materializedType.isConstantStorage(
        getASTContext(), /*ExcludeCtor*/ value, /*ExcludeDtor*/ false);

    type = mlir::cast<mlir::TypedAttr>(initialValue).getType();
  } else {
    // No initializer, the initialization will be provided when we initialize
    // the declaration which performed lifetime extension.
    type = getTypes().convertTypeForMem(materializedType);
  }

  // Create a global variable for this lifetime-extended temporary.
  cir::GlobalLinkageKind linkage = getCIRLinkageVarDefinition(varDecl, false);
  if (linkage == cir::GlobalLinkageKind::ExternalLinkage) {
    const VarDecl *initVD;
    if (varDecl->isStaticDataMember() && varDecl->getAnyInitializer(initVD) &&
        isa<CXXRecordDecl>(initVD->getLexicalDeclContext())) {
      // Temporaries defined inside a class get linkonce_odr linkage because the
      // calss can be defined in multiple translation units.
      llvm_unreachable("staticdatamember NYI");
    } else {
      // There is no need for this temporary to have external linkage if the
      // VarDecl has external linkage.
      linkage = cir::GlobalLinkageKind::InternalLinkage;
    }
  }
  auto targetAS = builder.getAddrSpaceAttr(addrSpace);

  auto loc = getLoc(expr->getSourceRange());
  auto gv = createGlobalOp(*this, loc, name, type, isConstant, targetAS,
                           nullptr, linkage);
  gv.setInitialValueAttr(initialValue);

  if (emitter)
    emitter->finalize(gv);
  // Don't assign dllimport or dllexport to lcoal linkage globals
  if (!gv.hasLocalLinkage()) {
    llvm_unreachable("NYI");
  }
  gv.setAlignment(align.getAsAlign().value());
  if (supportsCOMDAT() && gv.isWeakForLinker())
    llvm_unreachable("NYI");
  if (varDecl->getTLSKind())
    llvm_unreachable("NYI");
  mlir::Operation *cv = gv;
  if (addrSpace != LangAS::Default)
    llvm_unreachable("NYI");

  // Update the map with the new temporay. If we created a placeholder above,
  // replace it with the new global now.
  mlir::Operation *&entry = materializedGlobalTemporaryMap[expr];
  if (entry) {
    entry->replaceAllUsesWith(cv);
    entry->erase();
  }
  entry = cv;

  return cv;
}

// Emit code for a single top level declaration.
void CIRGenModule::emitTopLevelDecl(Decl *decl) {
  // Ignore dependent declarations
  if (decl->isTemplated())
    return;

  // Consteval function shouldn't be emitted.
  if (auto *FD = dyn_cast<FunctionDecl>(decl))
    if (FD->isConsteval())
      return;

  switch (decl->getKind()) {
  default:
    llvm::errs() << "emitTopLevelDecl codegen for decl kind '"
                 << decl->getDeclKindName() << "' not implemented\n";
    assert(false && "Not yet implemented");

  case Decl::TranslationUnit: {
    // This path is CIR only - CIRGen handles TUDecls because
    // of clang-tidy checks, that operate on TU granularity.
    TranslationUnitDecl *TU = cast<TranslationUnitDecl>(decl);
    for (DeclContext::decl_iterator D = TU->decls_begin(),
                                    DEnd = TU->decls_end();
         D != DEnd; ++D)
      emitTopLevelDecl(*D);
    return;
  }
  case Decl::Var:
  case Decl::Decomposition:
  case Decl::VarTemplateSpecialization:
    emitGlobal(cast<VarDecl>(decl));
    assert(!isa<DecompositionDecl>(decl) && "not implemented");
    // if (auto *DD = dyn_cast<DecompositionDecl>(decl))
    //   for (auto *B : DD->bindings())
    //     if (auto *HD = B->getHoldingVar())
    //       EmitGlobal(HD);
    break;

  case Decl::CXXConversion:
  case Decl::CXXMethod:
  case Decl::Function:
    emitGlobal(cast<FunctionDecl>(decl));
    assert(!codeGenOpts.CoverageMapping && "Coverage Mapping NYI");
    break;
  // C++ Decls
  case Decl::Namespace:
    emitDeclContext(cast<NamespaceDecl>(decl));
    break;
  case Decl::ClassTemplateSpecialization: {
    // const auto *Spec = cast<ClassTemplateSpecializationDecl>(decl);
    assert(!cir::MissingFeatures::generateDebugInfo() && "NYI");
  }
    [[fallthrough]];
  case Decl::CXXRecord: {
    CXXRecordDecl *crd = cast<CXXRecordDecl>(decl);
    // TODO: Handle debug info as CodeGenModule.cpp does
    for (auto *childDecl : crd->decls())
      if (isa<VarDecl>(childDecl) || isa<CXXRecordDecl>(childDecl))
        emitTopLevelDecl(childDecl);
    break;
  }
  // No code generation needed.
  case Decl::UsingShadow:
  case Decl::ClassTemplate:
  case Decl::VarTemplate:
  case Decl::Concept:
  case Decl::VarTemplatePartialSpecialization:
  case Decl::FunctionTemplate:
  case Decl::TypeAliasTemplate:
  case Decl::Block:
  case Decl::Empty:
  case Decl::Binding:
    break;
  case Decl::Using:     // using X; [C++]
  case Decl::UsingEnum: // using enum X; [C++]
  case Decl::NamespaceAlias:
  case Decl::UsingDirective: // using namespace X; [C++]
    assert(!cir::MissingFeatures::generateDebugInfo() && "NYI");
    break;
  case Decl::CXXConstructor:
    getCXXABI().emitCXXConstructors(cast<CXXConstructorDecl>(decl));
    break;
  case Decl::CXXDestructor:
    getCXXABI().emitCXXDestructors(cast<CXXDestructorDecl>(decl));
    break;

  case Decl::StaticAssert:
    // Nothing to do.
    break;

  case Decl::LinkageSpec:
    emitLinkageSpec(cast<LinkageSpecDecl>(decl));
    break;

  case Decl::Typedef:
  case Decl::TypeAlias: // using foo = bar; [C++11]
  case Decl::Record:
  case Decl::Enum:
    assert(!cir::MissingFeatures::generateDebugInfo() && "NYI");
    break;
  }
}

static bool shouldBeInCOMDAT(CIRGenModule &CGM, const Decl &D) {
  if (!CGM.supportsCOMDAT())
    return false;

  if (D.hasAttr<SelectAnyAttr>())
    return true;

  GVALinkage Linkage;
  if (auto *VD = dyn_cast<VarDecl>(&D))
    Linkage = CGM.getASTContext().GetGVALinkageForVariable(VD);
  else
    Linkage =
        CGM.getASTContext().GetGVALinkageForFunction(cast<FunctionDecl>(&D));

  switch (Linkage) {
  case clang::GVA_Internal:
  case clang::GVA_AvailableExternally:
  case clang::GVA_StrongExternal:
    return false;
  case clang::GVA_DiscardableODR:
  case clang::GVA_StrongODR:
    return true;
  }
  llvm_unreachable("No such linkage");
}

// TODO(cir): this could be a common method between LLVM codegen.
static bool isVarDeclStrongDefinition(const ASTContext &astContext,
                                      CIRGenModule &CGM, const VarDecl *D,
                                      bool NoCommon) {
  // Don't give variables common linkage if -fno-common was specified unless it
  // was overridden by a NoCommon attribute.
  if ((NoCommon || D->hasAttr<NoCommonAttr>()) && !D->hasAttr<CommonAttr>())
    return true;

  // C11 6.9.2/2:
  //   A declaration of an identifier for an object that has file scope without
  //   an initializer, and without a storage-class specifier or with the
  //   storage-class specifier static, constitutes a tentative definition.
  if (D->getInit() || D->hasExternalStorage())
    return true;

  // A variable cannot be both common and exist in a section.
  if (D->hasAttr<SectionAttr>())
    return true;

  // A variable cannot be both common and exist in a section.
  // We don't try to determine which is the right section in the front-end.
  // If no specialized section name is applicable, it will resort to default.
  if (D->hasAttr<PragmaClangBSSSectionAttr>() ||
      D->hasAttr<PragmaClangDataSectionAttr>() ||
      D->hasAttr<PragmaClangRelroSectionAttr>() ||
      D->hasAttr<PragmaClangRodataSectionAttr>())
    return true;

  // Thread local vars aren't considered common linkage.
  if (D->getTLSKind())
    return true;

  // Tentative definitions marked with WeakImportAttr are true definitions.
  if (D->hasAttr<WeakImportAttr>())
    return true;

  // A variable cannot be both common and exist in a comdat.
  if (shouldBeInCOMDAT(CGM, *D))
    return true;

  // Declarations with a required alignment do not have common linkage in MSVC
  // mode.
  if (astContext.getTargetInfo().getCXXABI().isMicrosoft()) {
    if (D->hasAttr<AlignedAttr>())
      return true;
    QualType VarType = D->getType();
    if (astContext.isAlignmentRequired(VarType))
      return true;

    if (const auto *RT = VarType->getAs<RecordType>()) {
      const RecordDecl *RD = RT->getDecl();
      for (const FieldDecl *FD : RD->fields()) {
        if (FD->isBitField())
          continue;
        if (FD->hasAttr<AlignedAttr>())
          return true;
        if (astContext.isAlignmentRequired(FD->getType()))
          return true;
      }
    }
  }

  // Microsoft's link.exe doesn't support alignments greater than 32 bytes for
  // common symbols, so symbols with greater alignment requirements cannot be
  // common.
  // Other COFF linkers (ld.bfd and LLD) support arbitrary power-of-two
  // alignments for common symbols via the aligncomm directive, so this
  // restriction only applies to MSVC environments.
  if (astContext.getTargetInfo().getTriple().isKnownWindowsMSVCEnvironment() &&
      astContext.getTypeAlignIfKnown(D->getType()) >
          astContext.toBits(CharUnits::fromQuantity(32)))
    return true;

  return false;
}

void CIRGenModule::setInitializer(cir::GlobalOp &global,
                                  mlir::Attribute value) {
  // Recompute visibility when updating initializer.
  global.setInitialValueAttr(value);
  mlir::SymbolTable::setSymbolVisibility(
      global, CIRGenModule::getMLIRVisibility(global));
}

mlir::SymbolTable::Visibility
CIRGenModule::getMLIRVisibility(cir::GlobalOp op) {
  // MLIR doesn't accept public symbols declarations (only
  // definitions).
  if (op.isDeclaration())
    return mlir::SymbolTable::Visibility::Private;
  return getMLIRVisibilityFromCIRLinkage(op.getLinkage());
}

mlir::SymbolTable::Visibility
CIRGenModule::getMLIRVisibilityFromCIRLinkage(cir::GlobalLinkageKind GLK) {
  switch (GLK) {
  case cir::GlobalLinkageKind::InternalLinkage:
  case cir::GlobalLinkageKind::PrivateLinkage:
    return mlir::SymbolTable::Visibility::Private;
  case cir::GlobalLinkageKind::ExternalLinkage:
  case cir::GlobalLinkageKind::ExternalWeakLinkage:
  case cir::GlobalLinkageKind::LinkOnceODRLinkage:
  case cir::GlobalLinkageKind::AvailableExternallyLinkage:
  case cir::GlobalLinkageKind::CommonLinkage:
  case cir::GlobalLinkageKind::WeakAnyLinkage:
  case cir::GlobalLinkageKind::WeakODRLinkage:
    return mlir::SymbolTable::Visibility::Public;
  default: {
    llvm::errs() << "visibility not implemented for '"
                 << stringifyGlobalLinkageKind(GLK) << "'\n";
    assert(0 && "not implemented");
  }
  }
  llvm_unreachable("linkage should be handled above!");
}

cir::VisibilityKind CIRGenModule::getGlobalVisibilityKindFromClangVisibility(
    clang::VisibilityAttr::VisibilityType visibility) {
  switch (visibility) {
  case clang::VisibilityAttr::VisibilityType::Default:
    return VisibilityKind::Default;
  case clang::VisibilityAttr::VisibilityType::Hidden:
    return VisibilityKind::Hidden;
  case clang::VisibilityAttr::VisibilityType::Protected:
    return VisibilityKind::Protected;
  }
  llvm_unreachable("unexpected visibility value");
}

cir::VisibilityAttr
CIRGenModule::getGlobalVisibilityAttrFromDecl(const Decl *decl) {
  const clang::VisibilityAttr *VA = decl->getAttr<clang::VisibilityAttr>();
  cir::VisibilityAttr cirVisibility =
      cir::VisibilityAttr::get(&getMLIRContext());
  if (VA) {
    cirVisibility = cir::VisibilityAttr::get(
        &getMLIRContext(),
        getGlobalVisibilityKindFromClangVisibility(VA->getVisibility()));
  }
  return cirVisibility;
}

cir::GlobalLinkageKind CIRGenModule::getCIRLinkageForDeclarator(
    const DeclaratorDecl *D, GVALinkage Linkage, bool IsConstantVariable) {
  if (Linkage == GVA_Internal)
    return cir::GlobalLinkageKind::InternalLinkage;

  if (D->hasAttr<WeakAttr>()) {
    if (IsConstantVariable)
      return cir::GlobalLinkageKind::WeakODRLinkage;
    else
      return cir::GlobalLinkageKind::WeakAnyLinkage;
  }

  if (const auto *FD = D->getAsFunction())
    if (FD->isMultiVersion() && Linkage == GVA_AvailableExternally)
      return cir::GlobalLinkageKind::LinkOnceAnyLinkage;

  // We are guaranteed to have a strong definition somewhere else,
  // so we can use available_externally linkage.
  if (Linkage == GVA_AvailableExternally)
    return cir::GlobalLinkageKind::AvailableExternallyLinkage;

  // Note that Apple's kernel linker doesn't support symbol
  // coalescing, so we need to avoid linkonce and weak linkages there.
  // Normally, this means we just map to internal, but for explicit
  // instantiations we'll map to external.

  // In C++, the compiler has to emit a definition in every translation unit
  // that references the function.  We should use linkonce_odr because
  // a) if all references in this translation unit are optimized away, we
  // don't need to codegen it.  b) if the function persists, it needs to be
  // merged with other definitions. c) C++ has the ODR, so we know the
  // definition is dependable.
  if (Linkage == GVA_DiscardableODR)
    return !astContext.getLangOpts().AppleKext
               ? cir::GlobalLinkageKind::LinkOnceODRLinkage
               : cir::GlobalLinkageKind::InternalLinkage;

  // An explicit instantiation of a template has weak linkage, since
  // explicit instantiations can occur in multiple translation units
  // and must all be equivalent. However, we are not allowed to
  // throw away these explicit instantiations.
  //
  // CUDA/HIP: For -fno-gpu-rdc case, device code is limited to one TU,
  // so say that CUDA templates are either external (for kernels) or internal.
  // This lets llvm perform aggressive inter-procedural optimizations. For
  // -fgpu-rdc case, device function calls across multiple TU's are allowed,
  // therefore we need to follow the normal linkage paradigm.
  if (Linkage == GVA_StrongODR) {
    if (getLangOpts().AppleKext)
      return cir::GlobalLinkageKind::ExternalLinkage;
    if (getLangOpts().CUDA && getLangOpts().CUDAIsDevice &&
        !getLangOpts().GPURelocatableDeviceCode)
      return D->hasAttr<CUDAGlobalAttr>()
                 ? cir::GlobalLinkageKind::ExternalLinkage
                 : cir::GlobalLinkageKind::InternalLinkage;
    return cir::GlobalLinkageKind::WeakODRLinkage;
  }

  // C++ doesn't have tentative definitions and thus cannot have common
  // linkage.
  if (!getLangOpts().CPlusPlus && isa<VarDecl>(D) &&
      !isVarDeclStrongDefinition(astContext, *this, cast<VarDecl>(D),
                                 getCodeGenOpts().NoCommon))
    return cir::GlobalLinkageKind::CommonLinkage;

  // selectany symbols are externally visible, so use weak instead of
  // linkonce.  MSVC optimizes away references to const selectany globals, so
  // all definitions should be the same and ODR linkage should be used.
  // http://msdn.microsoft.com/en-us/library/5tkz6s71.aspx
  if (D->hasAttr<SelectAnyAttr>())
    return cir::GlobalLinkageKind::WeakODRLinkage;

  // Otherwise, we have strong external linkage.
  assert(Linkage == GVA_StrongExternal);
  return cir::GlobalLinkageKind::ExternalLinkage;
}

/// This function is called when we implement a function with no prototype, e.g.
/// "int foo() {}". If there are existing call uses of the old function in the
/// module, this adjusts them to call the new function directly.
///
/// This is not just a cleanup: the always_inline pass requires direct calls to
/// functions to be able to inline them.  If there is a bitcast in the way, it
/// won't inline them. Instcombine normally deletes these calls, but it isn't
/// run at -O0.
void CIRGenModule::ReplaceUsesOfNonProtoTypeWithRealFunction(
    mlir::Operation *Old, cir::FuncOp NewFn) {

  // If we're redefining a global as a function, don't transform it.
  auto OldFn = dyn_cast<cir::FuncOp>(Old);
  if (!OldFn)
    return;

  // TODO(cir): this RAUW ignores the features below.
  assert(!cir::MissingFeatures::exceptions() && "Call vs Invoke NYI");
  assert(!cir::MissingFeatures::parameterAttributes());
  assert(!cir::MissingFeatures::operandBundles());
  assert(OldFn->getAttrs().size() > 1 && "Attribute forwarding NYI");

  // Mark new function as originated from a no-proto declaration.
  NewFn.setNoProtoAttr(OldFn.getNoProtoAttr());

  // Iterate through all calls of the no-proto function.
  auto SymUses = OldFn.getSymbolUses(OldFn->getParentOp());
  for (auto Use : SymUses.value()) {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto noProtoCallOp = dyn_cast<cir::CallOp>(Use.getUser())) {
      builder.setInsertionPoint(noProtoCallOp);

      // Patch call type with the real function type.
      auto realCallOp = builder.createCallOp(noProtoCallOp.getLoc(), NewFn,
                                             noProtoCallOp.getOperands());

      // Replace old no proto call with fixed call.
      noProtoCallOp.replaceAllUsesWith(realCallOp);
      noProtoCallOp.erase();
    } else if (auto getGlobalOp = dyn_cast<cir::GetGlobalOp>(Use.getUser())) {
      // Replace type
      getGlobalOp.getAddr().setType(
          cir::PointerType::get(&getMLIRContext(), NewFn.getFunctionType()));
    } else {
      llvm_unreachable("NIY");
    }
  }
}

cir::GlobalLinkageKind
CIRGenModule::getCIRLinkageVarDefinition(const VarDecl *VD, bool IsConstant) {
  assert(!IsConstant && "constant variables NYI");
  GVALinkage Linkage = astContext.GetGVALinkageForVariable(VD);
  return getCIRLinkageForDeclarator(VD, Linkage, IsConstant);
}

cir::GlobalLinkageKind CIRGenModule::getFunctionLinkage(GlobalDecl GD) {
  const auto *D = cast<FunctionDecl>(GD.getDecl());

  GVALinkage Linkage = astContext.GetGVALinkageForFunction(D);

  if (const auto *Dtor = dyn_cast<CXXDestructorDecl>(D))
    return getCXXABI().getCXXDestructorLinkage(Linkage, Dtor, GD.getDtorType());

  return getCIRLinkageForDeclarator(D, Linkage, /*IsConstantVariable=*/false);
}

void CIRGenModule::emitAliasForGlobal(StringRef mangledName,
                                      mlir::Operation *op, GlobalDecl aliasGD,
                                      cir::FuncOp aliasee,
                                      cir::GlobalLinkageKind linkage) {
  auto *aliasFD = dyn_cast<FunctionDecl>(aliasGD.getDecl());
  assert(aliasFD && "expected FunctionDecl");

  // The aliasee function type is different from the alias one, this difference
  // is specific to CIR because in LLVM the ptr types are already erased at this
  // point.
  auto &fnInfo = getTypes().arrangeCXXStructorDeclaration(aliasGD);
  auto fnType = getTypes().GetFunctionType(fnInfo);

  auto alias = createCIRFunction(getLoc(aliasGD.getDecl()->getSourceRange()),
                                 mangledName, fnType, aliasFD);
  alias.setAliasee(aliasee.getName());
  alias.setLinkage(linkage);
  // Declarations cannot have public MLIR visibility, just mark them private
  // but this really should have no meaning since CIR should not be using
  // this information to derive linkage information.
  mlir::SymbolTable::setSymbolVisibility(
      alias, mlir::SymbolTable::Visibility::Private);

  // Alias constructors and destructors are always unnamed_addr.
  assert(!cir::MissingFeatures::unnamedAddr());

  // Switch any previous uses to the alias.
  if (op) {
    llvm_unreachable("NYI");
  } else {
    // Name already set by createCIRFunction
  }

  // Finally, set up the alias with its proper name and attributes.
  setCommonAttributes(aliasGD, alias);
}

mlir::Type CIRGenModule::convertType(QualType type) {
  return genTypes.convertType(type);
}

bool CIRGenModule::verifyModule() {
  // Verify the module after we have finished constructing it, this will
  // check the structural properties of the IR and invoke any specific
  // verifiers we have on the CIR operations.
  return mlir::verify(theModule).succeeded();
}

std::pair<cir::FuncType, cir::FuncOp> CIRGenModule::getAddrAndTypeOfCXXStructor(
    GlobalDecl GD, const CIRGenFunctionInfo *FnInfo, cir::FuncType FnType,
    bool Dontdefer, ForDefinition_t IsForDefinition) {
  auto *MD = cast<CXXMethodDecl>(GD.getDecl());

  if (isa<CXXDestructorDecl>(MD)) {
    // Always alias equivalent complete destructors to base destructors in the
    // MS ABI.
    if (getTarget().getCXXABI().isMicrosoft() &&
        GD.getDtorType() == Dtor_Complete &&
        MD->getParent()->getNumVBases() == 0)
      llvm_unreachable("NYI");
  }

  if (!FnType) {
    if (!FnInfo)
      FnInfo = &getTypes().arrangeCXXStructorDeclaration(GD);
    FnType = getTypes().GetFunctionType(*FnInfo);
  }

  auto Fn = GetOrCreateCIRFunction(getMangledName(GD), FnType, GD,
                                   /*ForVtable=*/false, Dontdefer,
                                   /*IsThunk=*/false, IsForDefinition);

  return {FnType, Fn};
}

cir::FuncOp CIRGenModule::GetAddrOfFunction(clang::GlobalDecl GD, mlir::Type Ty,
                                            bool ForVTable, bool DontDefer,
                                            ForDefinition_t IsForDefinition) {
  assert(!cast<FunctionDecl>(GD.getDecl())->isConsteval() &&
         "consteval function should never be emitted");

  if (!Ty) {
    const auto *FD = cast<FunctionDecl>(GD.getDecl());
    Ty = convertType(FD->getType());
  }

  // Devirtualized destructor calls may come through here instead of via
  // getAddrOfCXXStructor. Make sure we use the MS ABI base destructor instead
  // of the complete destructor when necessary.
  if (const auto *DD = dyn_cast<CXXDestructorDecl>(GD.getDecl())) {
    if (getTarget().getCXXABI().isMicrosoft() &&
        GD.getDtorType() == Dtor_Complete &&
        DD->getParent()->getNumVBases() == 0)
      llvm_unreachable("NYI");
  }

  StringRef MangledName = getMangledName(GD);
  auto F = GetOrCreateCIRFunction(MangledName, Ty, GD, ForVTable, DontDefer,
                                  /*IsThunk=*/false, IsForDefinition);

  // As __global__ functions (kernels) always reside on device,
  // when we access them from host, we must refer to the kernel handle.
  // For HIP, we should never directly access the host device addr, but
  // instead the Global Variable of that stub. For CUDA, it's just the device
  // stub. For HIP, it's something different.
  if ((langOpts.HIP || langOpts.CUDA) && !langOpts.CUDAIsDevice &&
      cast<FunctionDecl>(GD.getDecl())->hasAttr<CUDAGlobalAttr>()) {
    auto *stubHandle = getCUDARuntime().getKernelHandle(F, GD);
    if (IsForDefinition)
      return F;

    if (langOpts.HIP)
      llvm_unreachable("NYI");
  }

  return F;
}

// Returns true if GD is a function decl with internal linkage and needs a
// unique suffix after the mangled name.
static bool isUniqueInternalLinkageDecl(GlobalDecl GD, CIRGenModule &CGM) {
  assert(CGM.getModuleNameHash().empty() &&
         "Unique internal linkage names NYI");

  return false;
}

static std::string getMangledNameImpl(CIRGenModule &CGM, GlobalDecl GD,
                                      const NamedDecl *ND,
                                      bool OmitMultiVersionMangling = false) {
  assert(!OmitMultiVersionMangling && "NYI");

  SmallString<256> Buffer;

  llvm::raw_svector_ostream Out(Buffer);
  MangleContext &MC = CGM.getCXXABI().getMangleContext();

  assert(CGM.getModuleNameHash().empty() && "NYI");
  auto ShouldMangle = MC.shouldMangleDeclName(ND);

  if (ShouldMangle) {
    MC.mangleName(GD.getWithDecl(ND), Out);
  } else {
    auto *II = ND->getIdentifier();
    assert(II && "Attempt to mangle unnamed decl.");

    const auto *FD = dyn_cast<FunctionDecl>(ND);

    if (FD &&
        FD->getType()->castAs<FunctionType>()->getCallConv() == CC_X86RegCall) {
      assert(0 && "NYI");
    } else if (FD && FD->hasAttr<CUDAGlobalAttr>() &&
               GD.getKernelReferenceKind() == KernelReferenceKind::Stub) {
      Out << "__device_stub__" << II->getName();
    } else {
      Out << II->getName();
    }
  }

  // Check if the module name hash should be appended for internal linkage
  // symbols. This should come before multi-version target suffixes are
  // appendded. This is to keep the name and module hash suffix of the internal
  // linkage function together. The unique suffix should only be added when name
  // mangling is done to make sure that the final name can be properly
  // demangled. For example, for C functions without prototypes, name mangling
  // is not done and the unique suffix should not be appended then.
  assert(!isUniqueInternalLinkageDecl(GD, CGM) && "NYI");

  if (const auto *FD = dyn_cast<FunctionDecl>(ND)) {
    assert(!FD->isMultiVersion() && "NYI");
  }
  assert(!CGM.getLangOpts().GPURelocatableDeviceCode && "NYI");

  return std::string(Out.str());
}

StringRef CIRGenModule::getMangledName(GlobalDecl GD) {
  auto CanonicalGD = GD.getCanonicalDecl();

  // Some ABIs don't have constructor variants. Make sure that base and complete
  // constructors get mangled the same.
  if (const auto *CD = dyn_cast<CXXConstructorDecl>(CanonicalGD.getDecl())) {
    if (!getTarget().getCXXABI().hasConstructorVariants()) {
      assert(false && "NYI");
    }
  }

  // Keep the first result in the case of a mangling collision.
  const auto *ND = cast<NamedDecl>(GD.getDecl());
  std::string MangledName = getMangledNameImpl(*this, GD, ND);

  auto Result = Manglings.insert(std::make_pair(MangledName, GD));
  return MangledDeclNames[CanonicalGD] = Result.first->first();
}

void CIRGenModule::emitTentativeDefinition(const VarDecl *D) {
  assert(!D->getInit() && "Cannot emit definite definitions here!");

  StringRef MangledName = getMangledName(D);
  auto *GV = getGlobalValue(MangledName);

  // TODO(cir): can a tentative definition come from something other than a
  // global op? If not, the assertion below is wrong and should be removed. If
  // so, getGlobalValue might be better of returining a global value interface
  // that alows use to manage different globals value types transparently.
  if (GV)
    assert(isa<cir::GlobalOp>(GV) &&
           "tentative definition can only be built from a cir.global_op");

  // We already have a definition, not declaration, with the same mangled name.
  // Emitting of declaration is not required (and actually overwrites emitted
  // definition).
  if (GV && !dyn_cast<cir::GlobalOp>(GV).isDeclaration())
    return;

  // If we have not seen a reference to this variable yet, place it into the
  // deferred declarations table to be emitted if needed later.
  if (!MustBeEmitted(D) && !GV) {
    DeferredDecls[MangledName] = D;
    return;
  }

  // The tentative definition is the only definition.
  emitGlobalVarDefinition(D);
}

void CIRGenModule::setGlobalVisibility(mlir::Operation *GV,
                                       const NamedDecl *D) const {
  assert(!cir::MissingFeatures::setGlobalVisibility());
}

void CIRGenModule::setDSOLocal(mlir::Operation *Op) const {
  assert(!cir::MissingFeatures::setDSOLocal());
  if (auto globalValue = dyn_cast<cir::CIRGlobalValueInterface>(Op)) {
    setDSOLocal(globalValue);
  }
}

void CIRGenModule::setGVProperties(mlir::Operation *Op,
                                   const NamedDecl *D) const {
  assert(!cir::MissingFeatures::setDLLImportDLLExport());
  setGVPropertiesAux(Op, D);
}

void CIRGenModule::setGVPropertiesAux(mlir::Operation *Op,
                                      const NamedDecl *D) const {
  setGlobalVisibility(Op, D);
  setDSOLocal(Op);
  assert(!cir::MissingFeatures::setPartition());
}

bool CIRGenModule::lookupRepresentativeDecl(StringRef MangledName,
                                            GlobalDecl &Result) const {
  auto Res = Manglings.find(MangledName);
  if (Res == Manglings.end())
    return false;
  Result = Res->getValue();
  return true;
}

cir::FuncOp CIRGenModule::createCIRFunction(mlir::Location loc, StringRef name,
                                            cir::FuncType Ty,
                                            const clang::FunctionDecl *FD) {
  // At the point we need to create the function, the insertion point
  // could be anywhere (e.g. callsite). Do not rely on whatever it might
  // be, properly save, find the appropriate place and restore.
  FuncOp f;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Some global emissions are triggered while emitting a function, e.g.
    // void s() { x.method() }
    //
    // Be sure to insert a new function before a current one.
    auto *curCGF = getCurrCIRGenFun();
    if (curCGF)
      builder.setInsertionPoint(curCGF->CurFn);

    f = builder.create<cir::FuncOp>(loc, name, Ty);

    if (FD)
      f.setAstAttr(makeFuncDeclAttr(FD, &getMLIRContext()));

    if (FD && !FD->hasPrototype())
      f.setNoProtoAttr(builder.getUnitAttr());

    assert(f.isDeclaration() && "expected empty body");

    // A declaration gets private visibility by default, but external linkage
    // as the default linkage.
    f.setLinkageAttr(cir::GlobalLinkageKindAttr::get(
        &getMLIRContext(), cir::GlobalLinkageKind::ExternalLinkage));
    mlir::SymbolTable::setSymbolVisibility(
        f, mlir::SymbolTable::Visibility::Private);

    // Initialize with empty dict of extra attributes.
    f.setExtraAttrsAttr(cir::ExtraFuncAttributesAttr::get(
        &getMLIRContext(), builder.getDictionaryAttr({})));

    if (!curCGF)
      theModule.push_back(f);
  }
  return f;
}

cir::FuncOp CIRGenModule::createRuntimeFunction(cir::FuncType Ty,
                                                StringRef Name, mlir::ArrayAttr,
                                                [[maybe_unused]] bool Local,
                                                bool AssumeConvergent) {
  if (AssumeConvergent) {
    llvm_unreachable("NYI");
  }
  if (Local)
    llvm_unreachable("NYI");

  auto entry = GetOrCreateCIRFunction(Name, Ty, GlobalDecl(),
                                      /*ForVtable=*/false);

  // Traditional codegen checks for a valid dyn_cast llvm::Function for `entry`,
  // no testcase that cover this path just yet though.
  if (!entry) {
    // Setup runtime CC, DLL support for windows and set dso local.
    llvm_unreachable("NYI");
  }

  return entry;
}

bool isDefaultedMethod(const clang::FunctionDecl *FD) {
  if (FD->isDefaulted() && isa<CXXMethodDecl>(FD) &&
      (cast<CXXMethodDecl>(FD)->isCopyAssignmentOperator() ||
       cast<CXXMethodDecl>(FD)->isMoveAssignmentOperator()))
    return true;
  return false;
}

mlir::Location CIRGenModule::getLocForFunction(const clang::FunctionDecl *FD) {
  bool invalidLoc = !FD || (FD->getSourceRange().getBegin().isInvalid() ||
                            FD->getSourceRange().getEnd().isInvalid());
  if (!invalidLoc)
    return getLoc(FD->getSourceRange());

  // Use the module location
  return theModule->getLoc();
}

/// Determines whether the language options require us to model
/// unwind exceptions.  We treat -fexceptions as mandating this
/// except under the fragile ObjC ABI with only ObjC exceptions
/// enabled.  This means, for example, that C with -fexceptions
/// enables this.
/// TODO(cir): can be shared with traditional LLVM codegen.
static bool hasUnwindExceptions(const LangOptions &LangOpts) {
  // If exceptions are completely disabled, obviously this is false.
  if (!LangOpts.Exceptions)
    return false;

  // If C++ exceptions are enabled, this is true.
  if (LangOpts.CXXExceptions)
    return true;

  // If ObjC exceptions are enabled, this depends on the ABI.
  if (LangOpts.ObjCExceptions) {
    return LangOpts.ObjCRuntime.hasUnwindExceptions();
  }

  return true;
}

void CIRGenModule::setCIRFunctionAttributesForDefinition(const Decl *decl,
                                                         FuncOp f) {
  mlir::NamedAttrList attrs{f.getExtraAttrs().getElements().getValue()};

  if ((!decl || !decl->hasAttr<NoUwtableAttr>()) && codeGenOpts.UnwindTables) {
    auto attr = cir::UWTableAttr::get(
        &getMLIRContext(), cir::UWTableKind(codeGenOpts.UnwindTables));
    attrs.set(attr.getMnemonic(), attr);
  }

  if (codeGenOpts.StackClashProtector)
    llvm_unreachable("NYI");

  if (codeGenOpts.StackProbeSize && codeGenOpts.StackProbeSize != 4096)
    llvm_unreachable("NYI");

  if (!hasUnwindExceptions(getLangOpts())) {
    auto attr = cir::NoThrowAttr::get(&getMLIRContext());
    attrs.set(attr.getMnemonic(), attr);
  }

  assert(!MissingFeatures::stackProtector());

  auto existingInlineAttr = dyn_cast_if_present<cir::InlineAttr>(
      attrs.get(cir::InlineAttr::getMnemonic()));
  bool isNoInline = existingInlineAttr && existingInlineAttr.isNoInline();
  bool isAlwaysInline =
      existingInlineAttr && existingInlineAttr.isAlwaysInline();

  if (!decl) {
    // Non-entry HLSL functions must always be inlined.
    if (getLangOpts().HLSL && !isNoInline) {
      auto attr = cir::InlineAttr::get(&getMLIRContext(),
                                       cir::InlineKind::AlwaysInline);
      attrs.set(attr.getMnemonic(), attr);
    } else if (!isAlwaysInline && codeGenOpts.getInlining() ==
                                      CodeGenOptions::OnlyAlwaysInlining) {
      // If we don't have a declaration to control inlining, the function isn't
      // explicitly marked as alwaysinline for semantic reasons, and inlining is
      // disabled, mark the function as noinline.
      auto attr =
          cir::InlineAttr::get(&getMLIRContext(), cir::InlineKind::NoInline);
      attrs.set(attr.getMnemonic(), attr);
    }

    f.setExtraAttrsAttr(cir::ExtraFuncAttributesAttr::get(
        &getMLIRContext(), attrs.getDictionary(&getMLIRContext())));
    return;
  }

  // Handle SME attributes that apply to function definitions,
  // rather than to function prototypes.
  if (decl->hasAttr<ArmLocallyStreamingAttr>())
    llvm_unreachable("NYI");

  if (auto *attr = decl->getAttr<ArmNewAttr>()) {
    if (attr->isNewZA())
      llvm_unreachable("NYI");
    if (attr->isNewZT0())
      llvm_unreachable("NYI");
  }

  // Track whether we need to add the optnone attribute,
  // starting with the default for this optimization level.
  bool shouldAddOptNone =
      !codeGenOpts.DisableO0ImplyOptNone && codeGenOpts.OptimizationLevel == 0;
  // We can't add optnone in the following cases, it won't pass the verifier.
  shouldAddOptNone &= !decl->hasAttr<MinSizeAttr>();
  shouldAddOptNone &= !decl->hasAttr<AlwaysInlineAttr>();

  // Non-entry HLSL functions must always be inlined.
  if (getLangOpts().HLSL && !isNoInline && !decl->hasAttr<NoInlineAttr>()) {
    auto attr =
        cir::InlineAttr::get(&getMLIRContext(), cir::InlineKind::AlwaysInline);
    attrs.set(attr.getMnemonic(), attr);
  } else if ((shouldAddOptNone || decl->hasAttr<OptimizeNoneAttr>()) &&
             !isAlwaysInline) {
    // Add optnone, but do so only if the function isn't always_inline.
    auto optNoneAttr = cir::OptNoneAttr::get(&getMLIRContext());
    attrs.set(optNoneAttr.getMnemonic(), optNoneAttr);

    // OptimizeNone implies noinline; we should not be inlining such functions.
    auto noInlineAttr =
        cir::InlineAttr::get(&getMLIRContext(), cir::InlineKind::NoInline);
    attrs.set(noInlineAttr.getMnemonic(), noInlineAttr);

    // We still need to handle naked functions even though optnone subsumes
    // much of their semantics.
    if (decl->hasAttr<NakedAttr>())
      llvm_unreachable("NYI");

    // OptimizeNone wins over OptimizeForSize and MinSize.
    assert(!MissingFeatures::optimizeForSize());
    assert(!MissingFeatures::minSize());
  } else if (decl->hasAttr<NakedAttr>()) {
    // Naked implies noinline: we should not be inlining such functions.
    llvm_unreachable("NYI");
  } else if (decl->hasAttr<NoDuplicateAttr>()) {
    llvm_unreachable("NYI");
  } else if (decl->hasAttr<NoInlineAttr>() && !isAlwaysInline) {
    // Add noinline if the function isn't always_inline.
    auto attr =
        cir::InlineAttr::get(&getMLIRContext(), cir::InlineKind::NoInline);
    attrs.set(attr.getMnemonic(), attr);
  } else if (decl->hasAttr<AlwaysInlineAttr>() && !isNoInline) {
    // (noinline wins over always_inline, and we can't specify both in IR)
    auto attr =
        cir::InlineAttr::get(&getMLIRContext(), cir::InlineKind::AlwaysInline);
    attrs.set(attr.getMnemonic(), attr);
  } else if (codeGenOpts.getInlining() == CodeGenOptions::OnlyAlwaysInlining) {
    // If we're not inlining, then force everything that isn't always_inline
    // to carry an explicit noinline attribute.
    if (!isAlwaysInline) {
      auto attr =
          cir::InlineAttr::get(&getMLIRContext(), cir::InlineKind::NoInline);
      attrs.set(attr.getMnemonic(), attr);
    }
  } else {
    // Otherwise, propagate the inline hint attribute and potentially use its
    // absence to mark things as noinline.
    // Search function and template pattern redeclarations for inline.
    if (auto *fd = dyn_cast<FunctionDecl>(decl)) {
      auto checkForInline = [](const FunctionDecl *decl) {
        auto checkRedeclForInline = [](const FunctionDecl *redecl) {
          return redecl->isInlineSpecified();
        };
        if (any_of(decl->redecls(), checkRedeclForInline))
          return true;
        const FunctionDecl *pattern = decl->getTemplateInstantiationPattern();
        if (!pattern)
          return false;
        return any_of(pattern->redecls(), checkRedeclForInline);
      };
      if (checkForInline(fd)) {
        auto attr = cir::InlineAttr::get(&getMLIRContext(),
                                         cir::InlineKind::InlineHint);
        attrs.set(attr.getMnemonic(), attr);
      } else if (codeGenOpts.getInlining() ==
                     CodeGenOptions::OnlyHintInlining &&
                 !fd->isInlined() && !isAlwaysInline) {
        auto attr =
            cir::InlineAttr::get(&getMLIRContext(), cir::InlineKind::NoInline);
        attrs.set(attr.getMnemonic(), attr);
      }
    }
  }

  // Add other optimization related attributes if we are optimizing this
  // function.
  if (!decl->hasAttr<OptimizeNoneAttr>()) {
    if (decl->hasAttr<ColdAttr>()) {
      llvm_unreachable("NYI");
    }
    if (decl->hasAttr<HotAttr>())
      llvm_unreachable("NYI");
    if (decl->hasAttr<MinSizeAttr>())
      assert(!MissingFeatures::minSize());
  }

  f.setExtraAttrsAttr(cir::ExtraFuncAttributesAttr::get(
      &getMLIRContext(), attrs.getDictionary(&getMLIRContext())));

  assert(!MissingFeatures::setFunctionAlignment());

  // In the cross-dso CFI mode with canonical jump tables, we want !type
  // attributes on definitions only.
  if (codeGenOpts.SanitizeCfiCrossDso &&
      codeGenOpts.SanitizeCfiCanonicalJumpTables) {
    llvm_unreachable("NYI");
  }

  assert(!MissingFeatures::memberFunctionPointerTypeMetadata());
}

void CIRGenModule::setCIRFunctionAttributes(GlobalDecl GD,
                                            const CIRGenFunctionInfo &info,
                                            cir::FuncOp func, bool isThunk) {
  // TODO(cir): More logic of constructAttributeList is needed.
  cir::CallingConv callingConv;
  cir::SideEffect sideEffect;

  // Initialize PAL with existing attributes to merge attributes.
  mlir::NamedAttrList PAL{func.getExtraAttrs().getElements().getValue()};
  constructAttributeList(func.getName(), info, GD, PAL, callingConv, sideEffect,
                         /*AttrOnCallSite=*/false, isThunk);
  func.setExtraAttrsAttr(cir::ExtraFuncAttributesAttr::get(
      &getMLIRContext(), PAL.getDictionary(&getMLIRContext())));

  // TODO(cir): Check X86_VectorCall incompatibility with WinARM64EC

  func.setCallingConv(callingConv);
}

void CIRGenModule::setFunctionAttributes(GlobalDecl globalDecl,
                                         cir::FuncOp func,
                                         bool isIncompleteFunction,
                                         bool isThunk) {
  // NOTE(cir): Original CodeGen checks if this is an intrinsic. In CIR we
  // represent them in dedicated ops. The correct attributes are ensured during
  // translation to LLVM. Thus, we don't need to check for them here.

  if (!isIncompleteFunction) {
    setCIRFunctionAttributes(globalDecl,
                             getTypes().arrangeGlobalDeclaration(globalDecl),
                             func, isThunk);
  }

  // TODO(cir): Complete the remaining part of the function.
  assert(!cir::MissingFeatures::setFunctionAttributes());

  // TODO(cir): This needs a lot of work to better match CodeGen. That
  // ultimately ends up in setGlobalVisibility, which already has the linkage of
  // the LLVM GV (corresponding to our FuncOp) computed, so it doesn't have to
  // recompute it here. This is a minimal fix for now.
  if (!isLocalLinkage(getFunctionLinkage(globalDecl))) {
    auto decl = globalDecl.getDecl();
    func.setGlobalVisibilityAttr(getGlobalVisibilityAttrFromDecl(decl));
  }
}

/// If the specified mangled name is not in the module,
/// create and return a CIR Function with the specified type. If there is
/// something in the module with the specified name, return it potentially
/// bitcasted to the right type.
///
/// If D is non-null, it specifies a decl that corresponded to this. This is
/// used to set the attributes on the function when it is first created.
cir::FuncOp CIRGenModule::GetOrCreateCIRFunction(
    StringRef MangledName, mlir::Type Ty, GlobalDecl GD, bool ForVTable,
    bool DontDefer, bool IsThunk, ForDefinition_t IsForDefinition,
    mlir::ArrayAttr ExtraAttrs) {
  assert(!IsThunk && "NYI");

  const auto *D = GD.getDecl();

  // Any attempts to use a MultiVersion function should result in retrieving the
  // iFunc instead. Name mangling will handle the rest of the changes.
  if (const auto *FD = cast_or_null<FunctionDecl>(D)) {
    // For the device mark the function as one that should be emitted.
    if (getLangOpts().OpenMPIsTargetDevice && FD->isDefined() && !DontDefer &&
        !IsForDefinition) {
      assert(0 && "OpenMP target functions NYI");
    }
    if (FD->isMultiVersion())
      llvm_unreachable("NYI");
  }

  // Lookup the entry, lazily creating it if necessary.
  mlir::Operation *Entry = getGlobalValue(MangledName);
  if (Entry) {
    assert(isa<cir::FuncOp>(Entry) &&
           "not implemented, only supports FuncOp for now");

    if (WeakRefReferences.erase(Entry)) {
      llvm_unreachable("NYI");
    }

    // Handle dropped DLL attributes.
    if (D && !D->hasAttr<DLLImportAttr>() && !D->hasAttr<DLLExportAttr>()) {
      // TODO(CIR): Entry->setDLLStorageClass
      setDSOLocal(Entry);
    }

    // If there are two attempts to define the same mangled name, issue an
    // error.
    auto Fn = cast<cir::FuncOp>(Entry);
    if (IsForDefinition && Fn && !Fn.isDeclaration()) {
      GlobalDecl OtherGD;
      // CHeck that GD is not yet in DiagnosedConflictingDefinitions is required
      // to make sure that we issue and error only once.
      if (lookupRepresentativeDecl(MangledName, OtherGD) &&
          (GD.getCanonicalDecl().getDecl() !=
           OtherGD.getCanonicalDecl().getDecl()) &&
          DiagnosedConflictingDefinitions.insert(GD).second) {
        getDiags().Report(D->getLocation(), diag::err_duplicate_mangled_name)
            << MangledName;
        getDiags().Report(OtherGD.getDecl()->getLocation(),
                          diag::note_previous_definition);
      }
    }

    if (Fn && Fn.getFunctionType() == Ty) {
      return Fn;
    }

    if (!IsForDefinition) {
      return Fn;
    }

    // TODO: clang checks here if this is a llvm::GlobalAlias... how will we
    // support this?
  }

  // This function doesn't have a complete type (for example, the return type is
  // an incomplete struct). Use a fake type instead, and make sure not to try to
  // set attributes.
  bool IsIncompleteFunction = false;

  cir::FuncType FTy;
  if (mlir::isa<cir::FuncType>(Ty)) {
    FTy = mlir::cast<cir::FuncType>(Ty);
  } else {
    assert(false && "NYI");
    // FTy = mlir::FunctionType::get(VoidTy, false);
    IsIncompleteFunction = true;
  }

  auto *FD = llvm::cast_or_null<FunctionDecl>(D);

  // TODO: CodeGen includeds the linkage (ExternalLinkage) and only passes the
  // mangledname if Entry is nullptr
  auto F = createCIRFunction(getLocForFunction(FD), MangledName, FTy, FD);

  // If we already created a function with the same mangled name (but different
  // type) before, take its name and add it to the list of functions to be
  // replaced with F at the end of CodeGen.
  //
  // This happens if there is a prototype for a function (e.g. "int f()") and
  // then a definition of a different type (e.g. "int f(int x)").
  if (Entry) {

    // Fetch a generic symbol-defining operation and its uses.
    auto SymbolOp = dyn_cast<mlir::SymbolOpInterface>(Entry);
    assert(SymbolOp && "Expected a symbol-defining operation");

    // TODO(cir): When can this symbol be something other than a function?
    assert(isa<cir::FuncOp>(Entry) && "NYI");

    // This might be an implementation of a function without a prototype, in
    // which case, try to do special replacement of calls which match the new
    // prototype. The really key thing here is that we also potentially drop
    // arguments from the call site so as to make a direct call, which makes the
    // inliner happier and suppresses a number of optimizer warnings (!) about
    // dropping arguments.
    if (SymbolOp.getSymbolUses(SymbolOp->getParentOp())) {
      ReplaceUsesOfNonProtoTypeWithRealFunction(Entry, F);
    }

    // Obliterate no-proto declaration.
    Entry->erase();
  }

  if (D)
    setFunctionAttributes(GD, F, IsIncompleteFunction, IsThunk);
  if (ExtraAttrs) {
    llvm_unreachable("NYI");
  }

  if (!DontDefer) {
    // All MSVC dtors other than the base dtor are linkonce_odr and delegate to
    // each other bottoming out wiht the base dtor. Therefore we emit non-base
    // dtors on usage, even if there is no dtor definition in the TU.
    if (isa_and_nonnull<CXXDestructorDecl>(D) &&
        getCXXABI().useThunkForDtorVariant(cast<CXXDestructorDecl>(D),
                                           GD.getDtorType())) {
      llvm_unreachable("NYI"); // addDeferredDeclToEmit(GD);
    }

    // This is the first use or definition of a mangled name. If there is a
    // deferred decl with this name, remember that we need to emit it at the end
    // of the file.
    auto DDI = DeferredDecls.find(MangledName);
    if (DDI != DeferredDecls.end()) {
      // Move the potentially referenced deferred decl to the
      // DeferredDeclsToEmit list, and remove it from DeferredDecls (since we
      // don't need it anymore).
      addDeferredDeclToEmit(DDI->second);
      DeferredDecls.erase(DDI);

      // Otherwise, there are cases we have to worry about where we're using a
      // declaration for which we must emit a definition but where we might not
      // find a top-level definition.
      //   - member functions defined inline in their classes
      //   - friend functions defined inline in some class
      //   - special member functions with implicit definitions
      // If we ever change our AST traversal to walk into class methods, this
      // will be unnecessary.
      //
      // We also don't emit a definition for a function if it's going to be an
      // entry in a vtable, unless it's already marked as used.
    } else if (getLangOpts().CPlusPlus && D) {
      // Look for a declaration that's lexically in a record.
      for (const auto *FD = cast<FunctionDecl>(D)->getMostRecentDecl(); FD;
           FD = FD->getPreviousDecl()) {
        if (isa<CXXRecordDecl>(FD->getLexicalDeclContext())) {
          if (FD->doesThisDeclarationHaveABody()) {
            addDeferredDeclToEmit(GD.getWithDecl(FD));
            break;
          }
        }
      }
    }
  }

  if (!IsIncompleteFunction) {
    assert(F.getFunctionType() == Ty);
    return F;
  }

  // TODO(cir): Might need bitcast to different address space.
  assert(!cir::MissingFeatures::addressSpace());
  return F;
}

mlir::Location CIRGenModule::getLoc(SourceLocation SLoc) {
  assert(SLoc.isValid() && "expected valid source location");
  const SourceManager &SM = astContext.getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(SLoc);
  StringRef Filename = PLoc.getFilename();
  return mlir::FileLineColLoc::get(builder.getStringAttr(Filename),
                                   PLoc.getLine(), PLoc.getColumn());
}

mlir::Location CIRGenModule::getLoc(SourceRange SLoc) {
  assert(SLoc.isValid() && "expected valid source location");
  mlir::Location B = getLoc(SLoc.getBegin());
  mlir::Location E = getLoc(SLoc.getEnd());
  SmallVector<mlir::Location, 2> locs = {B, E};
  mlir::Attribute metadata;
  return mlir::FusedLoc::get(locs, metadata, &getMLIRContext());
}

mlir::Location CIRGenModule::getLoc(mlir::Location lhs, mlir::Location rhs) {
  SmallVector<mlir::Location, 2> locs = {lhs, rhs};
  mlir::Attribute metadata;
  return mlir::FusedLoc::get(locs, metadata, &getMLIRContext());
}

void CIRGenModule::emitGlobalDecl(clang::GlobalDecl &D) {
  // We should call GetAddrOfGlobal with IsForDefinition set to true in order
  // to get a Value with exactly the type we need, not something that might
  // have been created for another decl with the same mangled name but
  // different type.
  auto *Op = GetAddrOfGlobal(D, ForDefinition);

  // In case of different address spaces, we may still get a cast, even with
  // IsForDefinition equal to true. Query mangled names table to get
  // GlobalValue.
  if (!Op) {
    Op = getGlobalValue(getMangledName(D));
  }

  // In case of different address spaces, we may still get a cast, even with
  // IsForDefinition equal to true. Query mangled names table to get
  // GlobalValue.
  if (!Op)
    llvm_unreachable("Address spaces NYI");

  // Make sure getGlobalValue returned non-null.
  assert(Op);

  // Check to see if we've already emitted this. This is necessary for a
  // couple of reasons: first, decls can end up in deferred-decls queue
  // multiple times, and second, decls can end up with definitions in unusual
  // ways (e.g. by an extern inline function acquiring a strong function
  // redefinition). Just ignore those cases.
  // TODO: Not sure what to map this to for MLIR
  auto globalValueOp = Op;
  if (auto Gv = dyn_cast<cir::GetGlobalOp>(Op)) {
    auto *result =
        mlir::SymbolTable::lookupSymbolIn(getModule(), Gv.getNameAttr());
    globalValueOp = result;
  }

  if (auto cirGlobalValue =
          dyn_cast<cir::CIRGlobalValueInterface>(globalValueOp)) {
    if (!cirGlobalValue.isDeclaration())
      return;
  }

  // If this is OpenMP, check if it is legal to emit this global normally.
  if (getLangOpts().OpenMP && openMPRuntime &&
      openMPRuntime->emitTargetGlobal(D))
    return;

  // Otherwise, emit the definition and move on to the next one.
  emitGlobalDefinition(D, Op);
}

void CIRGenModule::emitDeferred(unsigned recursionLimit) {
  // Emit deferred declare target declarations
  if (getLangOpts().OpenMP && !getLangOpts().OpenMPSimd)
    getOpenMPRuntime().emitDeferredTargetDecls();

  // Emit code for any potentially referenced deferred decls. Since a previously
  // unused static decl may become used during the generation of code for a
  // static function, iterate until no changes are made.

  if (!DeferredVTables.empty()) {
    emitDeferredVTables();

    // Emitting a vtable doesn't directly cause more vtables to
    // become deferred, although it can cause functions to be
    // emitted that then need those vtables.
    assert(DeferredVTables.empty());
  }

  // Emit CUDA/HIP static device variables referenced by host code only. Note we
  // should not clear CUDADeviceVarODRUsedByHost since it is still needed for
  // further handling.
  if ((getLangOpts().CUDA || getLangOpts().HIP) && getLangOpts().CUDAIsDevice &&
      !getASTContext().CUDADeviceVarODRUsedByHost.empty()) {
    llvm_unreachable("NYI");
  }

  // Stop if we're out of both deferred vtables and deferred declarations.
  if (DeferredDeclsToEmit.empty())
    return;

  // Grab the list of decls to emit. If emitGlobalDefinition schedules more
  // work, it will not interfere with this.
  std::vector<GlobalDecl> CurDeclsToEmit;
  CurDeclsToEmit.swap(DeferredDeclsToEmit);
  if (recursionLimit == 0)
    return;
  recursionLimit--;

  for (auto &D : CurDeclsToEmit) {
    if (getCodeGenOpts().ClangIRSkipFunctionsFromSystemHeaders) {
      auto *decl = D.getDecl();
      assert(decl && "expected decl");
      if (astContext.getSourceManager().isInSystemHeader(decl->getLocation()))
        continue;
    }

    emitGlobalDecl(D);

    // If we found out that we need to emit more decls, do that recursively.
    // This has the advantage that the decls are emitted in a DFS and related
    // ones are close together, which is convenient for testing.
    if (!DeferredVTables.empty() || !DeferredDeclsToEmit.empty()) {
      emitDeferred(recursionLimit);
      assert(DeferredVTables.empty() && DeferredDeclsToEmit.empty());
    }
  }
}

mlir::IntegerAttr CIRGenModule::getSize(CharUnits size) {
  return builder.getSizeFromCharUnits(&getMLIRContext(), size);
}

mlir::Operation *
CIRGenModule::GetAddrOfGlobal(GlobalDecl GD, ForDefinition_t IsForDefinition) {
  const Decl *D = GD.getDecl();

  if (isa<CXXConstructorDecl>(D) || isa<CXXDestructorDecl>(D))
    return getAddrOfCXXStructor(GD, /*FnInfo=*/nullptr, /*FnType=*/nullptr,
                                /*DontDefer=*/false, IsForDefinition);

  if (isa<CXXMethodDecl>(D)) {
    auto FInfo =
        &getTypes().arrangeCXXMethodDeclaration(cast<CXXMethodDecl>(D));
    auto Ty = getTypes().GetFunctionType(*FInfo);
    return GetAddrOfFunction(GD, Ty, /*ForVTable=*/false,
                             /*DontDefer=*/false, IsForDefinition);
  }

  if (isa<FunctionDecl>(D)) {
    const CIRGenFunctionInfo &FI = getTypes().arrangeGlobalDeclaration(GD);
    auto Ty = getTypes().GetFunctionType(FI);
    return GetAddrOfFunction(GD, Ty, /*ForVTable=*/false,
                             /*DontDefer=*/false, IsForDefinition);
  }

  return getAddrOfGlobalVar(cast<VarDecl>(D), /*Ty=*/nullptr, IsForDefinition)
      .getDefiningOp();
}

void CIRGenModule::Release() {
  assert(!MissingFeatures::emitModuleInitializers());
  emitDeferred(getCodeGenOpts().ClangIRBuildDeferredThreshold);
  assert(!MissingFeatures::emittedDeferredDecls());
  assert(!MissingFeatures::emitVTablesOpportunistically());
  assert(!MissingFeatures::applyGlobalValReplacements());
  applyReplacements();
  assert(!MissingFeatures::emitMultiVersionFunctions());

  assert(!MissingFeatures::incrementalExtensions());

  assert(!MissingFeatures::emitCXXModuleInitFunc());
  emitCXXGlobalInitFunc();
  assert(!MissingFeatures::emitCXXGlobalCleanUpFunc());
  assert(!MissingFeatures::registerGlobalDtorsWithAtExit());
  assert(!MissingFeatures::emitCXXThreadLocalInitFunc());
  assert(!MissingFeatures::objCRuntime());
  assert(!MissingFeatures::openMPRuntime());
  assert(!MissingFeatures::pgoReader());
  assert(!MissingFeatures::emitCtorList()); // GlobalCtors, GlobalDtors
  emitGlobalAnnotations();
  assert(!MissingFeatures::emitStaticExternCAliases());
  assert(!MissingFeatures::checkAliases());
  assert(!MissingFeatures::emitDeferredUnusedCoverageMappings());
  assert(!MissingFeatures::cirGenPGO()); // setValueProfilingFlag,
                                         // setProfileVersion
  assert(!MissingFeatures::coverageMapping());
  if (getCodeGenOpts().SanitizeCfiCrossDso) {
    llvm_unreachable("NYI");
  }
  if (langOpts.Sanitize.has(SanitizerKind::KCFI))
    llvm_unreachable("NYI");
  assert(!MissingFeatures::emitAtAvailableLinkGuard());
  if (astContext.getTargetInfo().getTriple().isWasm())
    llvm_unreachable("NYI");

  if (getTriple().isSPIRV() && getTriple().getVendor() == llvm::Triple::AMD) {
    llvm_unreachable("NYI");
  }

  // Emit a global array containing all external kernels or device variables
  // used by host functions and mark it as used for CUDA/HIP. This is necessary
  // to get kernels or device variables in archives linked in even if these
  // kernels or device variables are only used in host functions.
  if (!astContext.CUDAExternalDeviceDeclODRUsedByHost.empty()) {
    llvm_unreachable("NYI");
  }

  assert(!MissingFeatures::emitLLVMUsed());
  assert(!MissingFeatures::sanStats());

  if (codeGenOpts.Autolink && (astContext.getLangOpts().Modules ||
                               !MissingFeatures::linkerOptionsMetadata())) {
    assert(!MissingFeatures::emitModuleLinkOptions());
  }

  // On ELF we pass the dependent library specifiers directly to the linker
  // without manipulating them. This is in contrast to other platforms where
  // they are mapped to a specific linker option by the compiler. This
  // difference is a result of the greater variety of ELF linkers and the fact
  // that ELF linkers tend to handle libraries in a more complicated fashion
  // than on other platforms. This forces us to defer handling the dependent
  // libs to the linker.
  //
  // CUDA/HIP device and host libraries are different. Currently there is no
  // way to differentiate dependent libraries for host or device. Existing
  // usage of #pragma comment(lib, *) is intended for host libraries on
  // Windows. Therefore emit llvm.dependent-libraries only for host.
  assert(!MissingFeatures::elfDependentLibraries());

  assert(!MissingFeatures::dwarfVersion());

  if (codeGenOpts.Dwarf64)
    llvm_unreachable("NYI");

  if (astContext.getLangOpts().SemanticInterposition)
    // Require various optimization to respect semantic interposition.
    llvm_unreachable("NYI");

  if (codeGenOpts.EmitCodeView) {
    // Indicate that we want CodeView in the metadata.
    llvm_unreachable("NYI");
  }
  if (codeGenOpts.CodeViewGHash) {
    llvm_unreachable("NYI");
  }
  if (codeGenOpts.ControlFlowGuard) {
    // Function ID tables and checks for Control Flow Guard (cfguard=2).
    llvm_unreachable("NYI");
  } else if (codeGenOpts.ControlFlowGuardNoChecks) {
    // Function ID tables for Control Flow Guard (cfguard=1).
    llvm_unreachable("NYI");
  }
  if (codeGenOpts.EHContGuard) {
    // Function ID tables for EH Continuation Guard.
    llvm_unreachable("NYI");
  }
  if (astContext.getLangOpts().Kernel) {
    // Note if we are compiling with /kernel.
    llvm_unreachable("NYI");
  }
  if (codeGenOpts.OptimizationLevel > 0 && codeGenOpts.StrictVTablePointers) {
    // We don't support LTO with 2 with different StrictVTablePointers
    // FIXME: we could support it by stripping all the information introduced
    // by StrictVTablePointers.
    llvm_unreachable("NYI");
  }
  if (getModuleDebugInfo())
    // We support a single version in the linked module. The LLVM
    // parser will drop debug info with a different version number
    // (and warn about it, too).
    llvm_unreachable("NYI");

  // We need to record the widths of enums and wchar_t, so that we can generate
  // the correct build attributes in the ARM backend. wchar_size is also used by
  // TargetLibraryInfo.
  assert(!MissingFeatures::wcharWidth());

  if (getTriple().isOSzOS()) {
    llvm_unreachable("NYI");
  }

  llvm::Triple t = astContext.getTargetInfo().getTriple();
  if (t.isARM() || t.isThumb()) {
    // The minimum width of an enum in bytes
    assert(!MissingFeatures::enumWidth());
  }

  if (t.isRISCV()) {
    llvm_unreachable("NYI");
  }

  if (codeGenOpts.SanitizeCfiCrossDso) {
    // Indicate that we want cross-DSO control flow integrity checks.
    llvm_unreachable("NYI");
  }

  if (codeGenOpts.WholeProgramVTables) {
    // Indicate whether VFE was enabled for this module, so that the
    // vcall_visibility metadata added under whole program vtables is handled
    // appropriately in the optimizer.
    llvm_unreachable("NYI");
  }

  if (langOpts.Sanitize.has(SanitizerKind::CFIICall)) {
    llvm_unreachable("NYI");
  }

  if (codeGenOpts.SanitizeCfiICallNormalizeIntegers) {
    llvm_unreachable("NYI");
  }

  if (langOpts.Sanitize.has(SanitizerKind::KCFI)) {
    llvm_unreachable("NYI");
  }

  if (codeGenOpts.CFProtectionReturn &&
      target.checkCFProtectionReturnSupported(getDiags())) {
    // Indicate that we want to instrument return control flow protection.
    llvm_unreachable("NYI");
  }

  if (codeGenOpts.CFProtectionBranch &&
      target.checkCFProtectionBranchSupported(getDiags())) {
    // Indicate that we want to instrument branch control flow protection.
    llvm_unreachable("NYI");
  }

  if (codeGenOpts.FunctionReturnThunks)
    llvm_unreachable("NYI");

  if (codeGenOpts.IndirectBranchCSPrefix)
    llvm_unreachable("NYI");

  // Add module metadata for return address signing (ignoring
  // non-leaf/all) and stack tagging. These are actually turned on by function
  // attributes, but we use module metadata to emit build attributes. This is
  // needed for LTO, where the function attributes are inside bitcode
  // serialised into a global variable by the time build attributes are
  // emitted, so we can't access them. LTO objects could be compiled with
  // different flags therefore module flags are set to "Min" behavior to achieve
  // the same end result of the normal build where e.g BTI is off if any object
  // doesn't support it.
  if (astContext.getTargetInfo().hasFeature("ptrauth") &&
      langOpts.getSignReturnAddressScope() !=
          LangOptions::SignReturnAddressScopeKind::None)
    llvm_unreachable("NYI");
  if (langOpts.Sanitize.has(SanitizerKind::MemtagStack))
    llvm_unreachable("NYI");

  if (t.isARM() || t.isThumb() || t.isAArch64()) {
    if (langOpts.BranchTargetEnforcement)
      llvm_unreachable("NYI");
    if (langOpts.BranchProtectionPAuthLR)
      llvm_unreachable("NYI");
    if (langOpts.GuardedControlStack)
      llvm_unreachable("NYI");
    if (langOpts.hasSignReturnAddress())
      llvm_unreachable("NYI");
    if (langOpts.isSignReturnAddressScopeAll())
      llvm_unreachable("NYI");
    if (!langOpts.isSignReturnAddressWithAKey())
      llvm_unreachable("NYI");

    if (langOpts.PointerAuthELFGOT)
      llvm_unreachable("NYI");

    if (getTriple().isOSLinux()) {
      assert(getTriple().isOSBinFormatELF());
      assert(!MissingFeatures::ptrAuth());
    }
  }

  if (codeGenOpts.StackClashProtector)
    llvm_unreachable("NYI");

  if (codeGenOpts.StackProbeSize && codeGenOpts.StackProbeSize != 4096)
    llvm_unreachable("NYI");

  if (!codeGenOpts.MemoryProfileOutput.empty()) {
    llvm_unreachable("NYI");
  }

  if (langOpts.EHAsynch)
    llvm_unreachable("NYI");

  // Indicate whether this Module was compiled with -fopenmp
  assert(!MissingFeatures::openMP());

  // Emit OpenCL specific module metadata: OpenCL/SPIR version.
  if (langOpts.CUDAIsDevice && getTriple().isSPIRV())
    llvm_unreachable("CUDA SPIR-V NYI");
  if (langOpts.OpenCL) {
    emitOpenCLMetadata();
    // Emit SPIR version.
    if (getTriple().isSPIR())
      llvm_unreachable("SPIR target NYI");
  }

  // HLSL related end of code gen work items.
  if (langOpts.HLSL)
    llvm_unreachable("NYI");

  if (uint32_t picLevel = astContext.getLangOpts().PICLevel) {
    assert(picLevel < 3 && "Invalid PIC Level");
    assert(!MissingFeatures::setPICLevel());
    if (astContext.getLangOpts().PIE)
      assert(!MissingFeatures::setPIELevel());
  }

  if (getCodeGenOpts().CodeModel.size() > 0) {
    unsigned cm = llvm::StringSwitch<unsigned>(getCodeGenOpts().CodeModel)
                      .Case("tiny", llvm::CodeModel::Tiny)
                      .Case("small", llvm::CodeModel::Small)
                      .Case("kernel", llvm::CodeModel::Kernel)
                      .Case("medium", llvm::CodeModel::Medium)
                      .Case("large", llvm::CodeModel::Large)
                      .Default(~0u);
    if (cm != ~0u) {
      llvm::CodeModel::Model codeModel =
          static_cast<llvm::CodeModel::Model>(cm);
      (void)codeModel;
      assert(!MissingFeatures::codeModel());

      if ((cm == llvm::CodeModel::Medium || cm == llvm::CodeModel::Large) &&
          astContext.getTargetInfo().getTriple().getArch() ==
              llvm::Triple::x86_64) {
        assert(!MissingFeatures::largeDataThreshold());
      }
    }
  }

  if (codeGenOpts.NoPLT)
    llvm_unreachable("NYI");
  assert(!MissingFeatures::directAccessExternalData());
  if (codeGenOpts.UnwindTables)
    theModule->setAttr(
        cir::CIRDialect::getUWTableAttrName(),
        cir::UWTableAttr::get(&getMLIRContext(),
                              cir::UWTableKind(codeGenOpts.UnwindTables)));

  switch (codeGenOpts.getFramePointer()) {
  case CodeGenOptions::FramePointerKind::None:
    // 0 ("none") is the default.
    break;
  case CodeGenOptions::FramePointerKind::Reserved:
    assert(!MissingFeatures::setFramePointer());
    break;
  case CodeGenOptions::FramePointerKind::NonLeaf:
    assert(!MissingFeatures::setFramePointer());
    break;
  case CodeGenOptions::FramePointerKind::All:
    assert(!MissingFeatures::setFramePointer());
    break;
  }

  assert(!MissingFeatures::simplifyPersonality());

  if (getCodeGenOpts().EmitDeclMetadata)
    llvm_unreachable("NYI");

  if (getCodeGenOpts().CoverageNotesFile.size() ||
      getCodeGenOpts().CoverageDataFile.size())
    llvm_unreachable("NYI");

  if (getModuleDebugInfo())
    llvm_unreachable("NYI");

  assert(!MissingFeatures::emitVersionIdentMetadata());

  if (!getCodeGenOpts().RecordCommandLine.empty())
    llvm_unreachable("NYI");

  if (!getCodeGenOpts().StackProtectorGuard.empty())
    llvm_unreachable("NYI");
  if (!getCodeGenOpts().StackProtectorGuardReg.empty())
    llvm_unreachable("NYI");
  if (!getCodeGenOpts().StackProtectorGuardSymbol.empty())
    llvm_unreachable("NYI");
  if (getCodeGenOpts().StackProtectorGuardOffset != INT_MAX)
    llvm_unreachable("NYI");
  if (getCodeGenOpts().StackAlignment)
    llvm_unreachable("NYI");
  if (getCodeGenOpts().SkipRaxSetup)
    llvm_unreachable("NYI");
  if (getLangOpts().RegCall4)
    llvm_unreachable("NYI");

  if (getASTContext().getTargetInfo().getMaxTLSAlign())
    llvm_unreachable("NYI");

  assert(!MissingFeatures::emitTargetGlobals());

  assert(!MissingFeatures::emitTargetMetadata());

  assert(!MissingFeatures::emitBackendOptionsMetadata());

  // If there is device offloading code embed it in the host now.
  assert(!MissingFeatures::embedObject());

  // Set visibility from DLL storage class
  // We do this at the end of LLVM IR generation; after any operation
  // that might affect the DLL storage class or the visibility, and
  // before anything that might act on these.
  assert(!MissingFeatures::setVisibilityFromDLLStorageClass());

  // Check the tail call symbols are truly undefined.
  if (getTriple().isPPC() && !MissingFeatures::mustTailCallUndefinedGlobals()) {
    llvm_unreachable("NYI");
  }
}

namespace {
// TODO(cir): This should be a common helper shared with CodeGen.
struct FunctionIsDirectlyRecursive
    : public ConstStmtVisitor<FunctionIsDirectlyRecursive, bool> {
  const StringRef name;
  const Builtin::Context &builtinCtx;
  FunctionIsDirectlyRecursive(StringRef name,
                              const Builtin::Context &builtinCtx)
      : name(name), builtinCtx(builtinCtx) {}

  bool VisitCallExpr(const CallExpr *expr) {
    const FunctionDecl *func = expr->getDirectCallee();
    if (!func)
      return false;
    AsmLabelAttr *attr = func->getAttr<AsmLabelAttr>();
    if (attr && name == attr->getLabel())
      return true;
    unsigned builtinId = func->getBuiltinID();
    if (!builtinId || !builtinCtx.isLibFunction(builtinId))
      return false;
    StringRef builtinName = builtinCtx.getName(builtinId);
    if (builtinName.starts_with("__builtin_") &&
        name == builtinName.slice(strlen("__builtin_"), StringRef::npos)) {
      return true;
    }
    return false;
  }

  bool VisitStmt(const Stmt *stmt) {
    for (const Stmt *child : stmt->children())
      if (child && this->Visit(child))
        return true;
    return false;
  }
};
} // namespace

// isTriviallyRecursive - Check if this function calls another
// decl that, because of the asm attribute or the other decl being a builtin,
// ends up pointing to itself.
// TODO(cir): This should be a common helper shared with CodeGen.
bool CIRGenModule::isTriviallyRecursive(const FunctionDecl *func) {
  StringRef name;
  if (getCXXABI().getMangleContext().shouldMangleDeclName(func)) {
    // asm labels are a special kind of mangling we have to support.
    AsmLabelAttr *attr = func->getAttr<AsmLabelAttr>();
    if (!attr)
      return false;
    name = attr->getLabel();
  } else {
    name = func->getName();
  }

  FunctionIsDirectlyRecursive walker(name, astContext.BuiltinInfo);
  const Stmt *body = func->getBody();
  return body ? walker.Visit(body) : false;
}

// TODO(cir): This should be a common helper shared with CodeGen.
bool CIRGenModule::shouldEmitFunction(GlobalDecl globalDecl) {
  if (getFunctionLinkage(globalDecl) !=
      GlobalLinkageKind::AvailableExternallyLinkage)
    return true;

  const auto *func = cast<FunctionDecl>(globalDecl.getDecl());
  // Inline builtins declaration must be emitted. They often are fortified
  // functions.
  if (func->isInlineBuiltinDeclaration())
    return true;

  if (codeGenOpts.OptimizationLevel == 0 && !func->hasAttr<AlwaysInlineAttr>())
    return false;

  // We don't import function bodies from other named module units since that
  // behavior may break ABI compatibility of the current unit.
  if (const Module *mod = func->getOwningModule();
      mod && mod->getTopLevelModule()->isNamedModule() &&
      astContext.getCurrentNamedModule() != mod->getTopLevelModule()) {
    // There are practices to mark template member function as always-inline
    // and mark the template as extern explicit instantiation but not give
    // the definition for member function. So we have to emit the function
    // from explicitly instantiation with always-inline.
    //
    // See https://github.com/llvm/llvm-project/issues/86893 for details.
    //
    // TODO: Maybe it is better to give it a warning if we call a non-inline
    // function from other module units which is marked as always-inline.
    if (!func->isTemplateInstantiation() || !func->hasAttr<AlwaysInlineAttr>())
      return false;
  }

  if (func->hasAttr<NoInlineAttr>())
    return false;

  if (func->hasAttr<DLLImportAttr>() && !func->hasAttr<AlwaysInlineAttr>())
    assert(!cir::MissingFeatures::setDLLImportDLLExport() &&
           "shouldEmitFunction for dllimport is NYI");

  // PR9614. Avoid cases where the source code is lying to us. An available
  // externally function should have an equivalent function somewhere else,
  // but a function that calls itself through asm label/`__builtin_` trickery is
  // clearly not equivalent to the real implementation.
  // This happens in glibc's btowc and in some configure checks.
  return !isTriviallyRecursive(func);
}

bool CIRGenModule::supportsCOMDAT() const {
  return getTriple().supportsCOMDAT();
}

void CIRGenModule::maybeSetTrivialComdat(const Decl &d, mlir::Operation *op) {
  if (!shouldBeInCOMDAT(*this, d))
    return;
  auto globalOp = dyn_cast_or_null<cir::GlobalOp>(op);
  if (globalOp)
    globalOp.setComdat(true);
  // Keep it as missing feature as we need to implement comdat for FuncOp.
  // in the future.
  assert(!cir::MissingFeatures::setComdat() && "NYI");
}

bool CIRGenModule::isInNoSanitizeList(SanitizerMask Kind, cir::FuncOp Fn,
                                      SourceLocation Loc) const {
  const auto &NoSanitizeL = getASTContext().getNoSanitizeList();
  // NoSanitize by function name.
  if (NoSanitizeL.containsFunction(Kind, Fn.getName()))
    llvm_unreachable("NYI");
  // NoSanitize by location.
  if (Loc.isValid())
    return NoSanitizeL.containsLocation(Kind, Loc);
  // If location is unknown, this may be a compiler-generated function. Assume
  // it's located in the main file.
  auto &SM = getASTContext().getSourceManager();
  FileEntryRef MainFile = *SM.getFileEntryRefForID(SM.getMainFileID());
  if (NoSanitizeL.containsFile(Kind, MainFile.getName()))
    return true;

  // Check "src" prefix.
  if (Loc.isValid())
    return NoSanitizeL.containsLocation(Kind, Loc);
  // If location is unknown, this may be a compiler-generated function. Assume
  // it's located in the main file.
  return NoSanitizeL.containsFile(Kind, MainFile.getName());
}

void CIRGenModule::AddDeferredUnusedCoverageMapping(Decl *D) {
  // Do we need to generate coverage mapping?
  if (!codeGenOpts.CoverageMapping)
    return;

  llvm_unreachable("NYI");
}

void CIRGenModule::UpdateCompletedType(const TagDecl *TD) {
  // Make sure that this type is translated.
  genTypes.UpdateCompletedType(TD);
}

void CIRGenModule::addReplacement(StringRef Name, mlir::Operation *Op) {
  Replacements[Name] = Op;
}

void CIRGenModule::replacePointerTypeArgs(cir::FuncOp OldF, cir::FuncOp NewF) {
  auto optionalUseRange = OldF.getSymbolUses(theModule);
  if (!optionalUseRange)
    return;

  for (auto U : *optionalUseRange) {
    // CallTryOp only shows up after FlattenCFG.
    auto Call = mlir::dyn_cast<cir::CallOp>(U.getUser());
    if (!Call)
      continue;

    auto ArgOps = Call.getArgOps();
    auto FuncArgTypes = NewF.getFunctionType().getInputs();
    for (unsigned I = 0; I < FuncArgTypes.size(); I++) {
      if (ArgOps[I].getType() == FuncArgTypes[I])
        continue;

      auto argPointerTy = mlir::dyn_cast<cir::PointerType>(ArgOps[I].getType());
      auto funcArgPointerTy = mlir::dyn_cast<cir::PointerType>(FuncArgTypes[I]);

      // If we can't solve it, leave it for the verifier to bail out.
      if (!argPointerTy || !funcArgPointerTy)
        continue;

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(Call);
      auto castedArg =
          builder.createBitcast(Call.getLoc(), ArgOps[I], funcArgPointerTy);
      Call.setArg(I, castedArg);
    }
  }
}

void CIRGenModule::applyReplacements() {
  for (auto &I : Replacements) {
    StringRef MangledName = I.first();
    mlir::Operation *Replacement = I.second;
    auto *Entry = getGlobalValue(MangledName);
    if (!Entry)
      continue;
    assert(isa<cir::FuncOp>(Entry) && "expected function");
    auto OldF = cast<cir::FuncOp>(Entry);
    auto NewF = dyn_cast<cir::FuncOp>(Replacement);
    assert(NewF && "not implemented");

    // LLVM has opaque pointer but CIR not. So we may have to handle these
    // different pointer types when performing replacement.
    replacePointerTypeArgs(OldF, NewF);

    // Replace old with new, but keep the old order.
    if (OldF.replaceAllSymbolUses(NewF.getSymNameAttr(), theModule).failed())
      llvm_unreachable("internal error, cannot RAUW symbol");
    if (NewF) {
      NewF->moveBefore(OldF);
      OldF->erase();
    }
  }
}

void CIRGenModule::emitExplicitCastExprType(const ExplicitCastExpr *E,
                                            CIRGenFunction *CGF) {
  // Bind VLAs in the cast type.
  if (CGF && E->getType()->isVariablyModifiedType())
    llvm_unreachable("NYI");

  assert(!cir::MissingFeatures::generateDebugInfo() && "NYI");
}

void CIRGenModule::HandleCXXStaticMemberVarInstantiation(VarDecl *VD) {
  auto DK = VD->isThisDeclarationADefinition();
  if (DK == VarDecl::Definition && VD->hasAttr<DLLImportAttr>())
    return;

  TemplateSpecializationKind TSK = VD->getTemplateSpecializationKind();
  // If we have a definition, this might be a deferred decl. If the
  // instantiation is explicit, make sure we emit it at the end.
  if (VD->getDefinition() && TSK == TSK_ExplicitInstantiationDefinition) {
    llvm_unreachable("NYI");
  }

  emitTopLevelDecl(VD);
}

cir::GlobalOp CIRGenModule::createOrReplaceCXXRuntimeVariable(
    mlir::Location loc, StringRef Name, mlir::Type Ty,
    cir::GlobalLinkageKind Linkage, clang::CharUnits Alignment) {
  cir::GlobalOp OldGV{};
  auto GV = dyn_cast_or_null<cir::GlobalOp>(
      mlir::SymbolTable::lookupSymbolIn(getModule(), Name));

  if (GV) {
    // Check if the variable has the right type.
    if (GV.getSymType() == Ty)
      return GV;

    // Because C++ name mangling, the only way we can end up with an already
    // existing global with the same name is if it has been declared extern
    // "C".
    assert(GV.isDeclaration() && "Declaration has wrong type!");
    OldGV = GV;
  }

  // Create a new variable.
  GV = CIRGenModule::createGlobalOp(*this, loc, Name, Ty);

  // Set up extra information and add to the module
  GV.setLinkageAttr(
      cir::GlobalLinkageKindAttr::get(&getMLIRContext(), Linkage));
  mlir::SymbolTable::setSymbolVisibility(GV,
                                         CIRGenModule::getMLIRVisibility(GV));

  if (OldGV) {
    // Replace occurrences of the old variable if needed.
    GV.setName(OldGV.getName());
    if (!OldGV->use_empty()) {
      // TODO(cir): remove erase call above and use replaceGlobal here.
      llvm_unreachable("NYI");
    }
    OldGV->erase();
  }

  if (supportsCOMDAT() && cir::isWeakForLinker(Linkage) &&
      !GV.hasAvailableExternallyLinkage()) {
    GV.setComdat(true);
  }

  GV.setAlignmentAttr(getSize(Alignment));
  setDSOLocal(static_cast<mlir::Operation *>(GV));
  return GV;
}

bool CIRGenModule::shouldOpportunisticallyEmitVTables() {
  if (codeGenOpts.OptimizationLevel != 0)
    llvm_unreachable("NYI");
  return codeGenOpts.OptimizationLevel > 0;
}

void CIRGenModule::emitVTableTypeMetadata(const CXXRecordDecl *RD,
                                          cir::GlobalOp VTable,
                                          const VTableLayout &VTLayout) {
  if (!getCodeGenOpts().LTOUnit)
    return;
  llvm_unreachable("NYI");
}

mlir::Attribute CIRGenModule::getAddrOfRTTIDescriptor(mlir::Location loc,
                                                      QualType Ty, bool ForEH) {
  // Return a bogus pointer if RTTI is disabled, unless it's for EH.
  // FIXME: should we even be calling this method if RTTI is disabled
  // and it's not for EH?
  if (!shouldEmitRTTI(ForEH))
    return getBuilder().getConstNullPtrAttr(builder.getUInt8PtrTy());

  if (ForEH && Ty->isObjCObjectPointerType() &&
      getLangOpts().ObjCRuntime.isGNUFamily()) {
    llvm_unreachable("NYI");
  }

  return getCXXABI().getAddrOfRTTIDescriptor(loc, Ty);
}

/// TODO(cir): once we have cir.module, add this as a convenience method there.
///
/// Look up the specified global in the module symbol table.
///   1. If it does not exist, add a declaration of the global and return it.
///   2. Else, the global exists but has the wrong type: return the function
///      with a constantexpr cast to the right type.
///   3. Finally, if the existing global is the correct declaration, return the
///      existing global.
cir::GlobalOp CIRGenModule::getOrInsertGlobal(
    mlir::Location loc, StringRef Name, mlir::Type Ty,
    llvm::function_ref<cir::GlobalOp()> CreateGlobalCallback) {
  // See if we have a definition for the specified global already.
  auto GV = dyn_cast_or_null<cir::GlobalOp>(getGlobalValue(Name));
  if (!GV) {
    GV = CreateGlobalCallback();
  }
  assert(GV && "The CreateGlobalCallback is expected to create a global");

  // If the variable exists but has the wrong type, return a bitcast to the
  // right type.
  auto GVTy = GV.getSymType();
  assert(!cir::MissingFeatures::addressSpace());
  auto PTy = builder.getPointerTo(Ty);

  if (GVTy != PTy)
    llvm_unreachable("NYI");

  // Otherwise, we just found the existing function or a prototype.
  return GV;
}

// Overload to construct a global variable using its constructor's defaults.
cir::GlobalOp CIRGenModule::getOrInsertGlobal(mlir::Location loc,
                                              StringRef Name, mlir::Type Ty) {
  return getOrInsertGlobal(loc, Name, Ty, [&] {
    return CIRGenModule::createGlobalOp(*this, loc, Name,
                                        builder.getPointerTo(Ty));
  });
}

// TODO(cir): this can be shared with LLVM codegen.
CharUnits CIRGenModule::computeNonVirtualBaseClassOffset(
    const CXXRecordDecl *DerivedClass, CastExpr::path_const_iterator Start,
    CastExpr::path_const_iterator End) {
  CharUnits Offset = CharUnits::Zero();

  const ASTContext &astContext = getASTContext();
  const CXXRecordDecl *RD = DerivedClass;

  for (CastExpr::path_const_iterator I = Start; I != End; ++I) {
    const CXXBaseSpecifier *Base = *I;
    assert(!Base->isVirtual() && "Should not see virtual bases here!");

    // Get the layout.
    const ASTRecordLayout &Layout = astContext.getASTRecordLayout(RD);

    const auto *BaseDecl =
        cast<CXXRecordDecl>(Base->getType()->castAs<RecordType>()->getDecl());

    // Add the offset.
    Offset += Layout.getBaseClassOffset(BaseDecl);

    RD = BaseDecl;
  }

  return Offset;
}

void CIRGenModule::Error(SourceLocation loc, StringRef message) {
  unsigned diagID = getDiags().getCustomDiagID(DiagnosticsEngine::Error, "%0");
  getDiags().Report(astContext.getFullLoc(loc), diagID) << message;
}

/// Print out an error that codegen doesn't support the specified stmt yet.
void CIRGenModule::ErrorUnsupported(const Stmt *S, const char *Type) {
  unsigned DiagID = getDiags().getCustomDiagID(DiagnosticsEngine::Error,
                                               "cannot compile this %0 yet");
  std::string Msg = Type;
  getDiags().Report(astContext.getFullLoc(S->getBeginLoc()), DiagID)
      << Msg << S->getSourceRange();
}

/// Print out an error that codegen doesn't support the specified decl yet.
void CIRGenModule::ErrorUnsupported(const Decl *D, const char *Type) {
  unsigned DiagID = getDiags().getCustomDiagID(DiagnosticsEngine::Error,
                                               "cannot compile this %0 yet");
  std::string Msg = Type;
  getDiags().Report(astContext.getFullLoc(D->getLocation()), DiagID) << Msg;
}

cir::SourceLanguage CIRGenModule::getCIRSourceLanguage() {
  using ClangStd = clang::LangStandard;
  using CIRLang = cir::SourceLanguage;
  auto opts = getLangOpts();

  if (opts.OpenCL && !opts.OpenCLCPlusPlus)
    return CIRLang::OpenCLC;

  if (opts.CPlusPlus || opts.CPlusPlus11 || opts.CPlusPlus14 ||
      opts.CPlusPlus17 || opts.CPlusPlus20 || opts.CPlusPlus23 ||
      opts.CPlusPlus26)
    return CIRLang::CXX;
  if (opts.C99 || opts.C11 || opts.C17 || opts.C23 ||
      opts.LangStd == ClangStd::lang_c89 ||
      opts.LangStd == ClangStd::lang_gnu89)
    return CIRLang::C;

  // TODO(cir): support remaining source languages.
  llvm_unreachable("CIR does not yet support the given source language");
}

LangAS CIRGenModule::getGlobalVarAddressSpace(const VarDecl *D) {
  if (langOpts.OpenCL) {
    LangAS AS = D ? D->getType().getAddressSpace() : LangAS::opencl_global;
    assert(AS == LangAS::opencl_global || AS == LangAS::opencl_global_device ||
           AS == LangAS::opencl_global_host || AS == LangAS::opencl_constant ||
           AS == LangAS::opencl_local || AS >= LangAS::FirstTargetAddressSpace);
    return AS;
  }

  if (langOpts.SYCLIsDevice &&
      (!D || D->getType().getAddressSpace() == LangAS::Default))
    llvm_unreachable("NYI");

  if (langOpts.CUDA && langOpts.CUDAIsDevice) {
    if (D) {
      if (D->hasAttr<CUDAConstantAttr>())
        return LangAS::cuda_constant;
      if (D->hasAttr<CUDASharedAttr>())
        return LangAS::cuda_shared;
      if (D->hasAttr<CUDADeviceAttr>())
        return LangAS::cuda_device;
      if (D->getType().isConstQualified())
        return LangAS::cuda_constant;
    }
    return LangAS::cuda_device;
  }

  if (langOpts.OpenMP)
    llvm_unreachable("NYI");

  return getTargetCIRGenInfo().getGlobalVarAddressSpace(*this, D);
}

mlir::ArrayAttr CIRGenModule::emitAnnotationArgs(const AnnotateAttr *attr) {
  ArrayRef<Expr *> exprs = {attr->args_begin(), attr->args_size()};
  if (exprs.empty()) {
    return mlir::ArrayAttr::get(&getMLIRContext(), {});
  }
  llvm::FoldingSetNodeID id;
  for (Expr *e : exprs) {
    id.Add(cast<clang::ConstantExpr>(e)->getAPValueResult());
  }
  mlir::ArrayAttr &lookup = annotationArgs[id.ComputeHash()];
  if (lookup)
    return lookup;

  llvm::SmallVector<mlir::Attribute, 4> args;
  args.reserve(exprs.size());
  for (Expr *e : exprs) {
    auto &ce = *cast<clang::ConstantExpr>(e);
    if (auto *const strE =
            clang::dyn_cast<clang::StringLiteral>(ce.IgnoreParenCasts())) {
      // Add trailing null character as StringLiteral->getString() does not
      args.push_back(builder.getStringAttr(strE->getString()));
    } else if (ce.hasAPValueResult()) {
      // Handle case which can be evaluated to some numbers, not only literals
      const auto &ap = ce.getAPValueResult();
      if (ap.isInt()) {
        args.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(&getMLIRContext(),
                                   ap.getInt().getBitWidth()),
            ap.getInt()));
      } else {
        llvm_unreachable("NYI like float, fixed-point, array...");
      }
    } else {
      llvm_unreachable("NYI");
    }
  }

  lookup = builder.getArrayAttr(args);
  return lookup;
}

cir::AnnotationAttr
CIRGenModule::emitAnnotateAttr(const clang::AnnotateAttr *aa) {
  mlir::StringAttr annoGV = builder.getStringAttr(aa->getAnnotation());
  mlir::ArrayAttr args = emitAnnotationArgs(aa);
  return cir::AnnotationAttr::get(&getMLIRContext(), annoGV, args);
}

void CIRGenModule::addGlobalAnnotations(const ValueDecl *d,
                                        mlir::Operation *gv) {
  assert(d->hasAttr<AnnotateAttr>() && "no annotate attribute");
  assert((isa<GlobalOp>(gv) || isa<FuncOp>(gv)) &&
         "annotation only on globals");
  llvm::SmallVector<mlir::Attribute, 4> annotations;
  for (auto *i : d->specific_attrs<AnnotateAttr>())
    annotations.push_back(emitAnnotateAttr(i));
  if (auto global = dyn_cast<cir::GlobalOp>(gv))
    global.setAnnotationsAttr(builder.getArrayAttr(annotations));
  else if (auto func = dyn_cast<cir::FuncOp>(gv))
    func.setAnnotationsAttr(builder.getArrayAttr(annotations));
}

void CIRGenModule::emitGlobalAnnotations() {
  for (const auto &[mangledName, vd] : deferredAnnotations) {
    mlir::Operation *gv = getGlobalValue(mangledName);
    if (gv)
      addGlobalAnnotations(vd, gv);
  }
  deferredAnnotations.clear();
}

cir::TBAAAttr CIRGenModule::getTBAATypeInfo(QualType QTy) {
  if (!tbaa) {
    return nullptr;
  }
  return tbaa->getTypeInfo(QTy);
}

TBAAAccessInfo CIRGenModule::getTBAAAccessInfo(QualType accessType) {
  if (!tbaa) {
    return TBAAAccessInfo();
  }
  if (getLangOpts().CUDAIsDevice) {
    llvm_unreachable("NYI");
  }
  return tbaa->getAccessInfo(accessType);
}

TBAAAccessInfo
CIRGenModule::getTBAAVTablePtrAccessInfo(mlir::Type VTablePtrType) {
  if (!tbaa)
    return TBAAAccessInfo();
  return tbaa->getVTablePtrAccessInfo(VTablePtrType);
}

mlir::ArrayAttr CIRGenModule::getTBAAStructInfo(QualType QTy) {
  if (!tbaa)
    return nullptr;
  return tbaa->getTBAAStructInfo(QTy);
}

cir::TBAAAttr CIRGenModule::getTBAABaseTypeInfo(QualType QTy) {
  if (!tbaa) {
    return nullptr;
  }
  return tbaa->getBaseTypeInfo(QTy);
}

cir::TBAAAttr CIRGenModule::getTBAAAccessTagInfo(TBAAAccessInfo tbaaInfo) {
  if (!tbaa) {
    return nullptr;
  }
  return tbaa->getAccessTagInfo(tbaaInfo);
}

TBAAAccessInfo CIRGenModule::mergeTBAAInfoForCast(TBAAAccessInfo SourceInfo,
                                                  TBAAAccessInfo TargetInfo) {
  if (!tbaa)
    return TBAAAccessInfo();
  return tbaa->mergeTBAAInfoForCast(SourceInfo, TargetInfo);
}

TBAAAccessInfo
CIRGenModule::mergeTBAAInfoForConditionalOperator(TBAAAccessInfo InfoA,
                                                  TBAAAccessInfo InfoB) {
  if (!tbaa)
    return TBAAAccessInfo();
  return tbaa->mergeTBAAInfoForConditionalOperator(InfoA, InfoB);
}

TBAAAccessInfo
CIRGenModule::mergeTBAAInfoForMemoryTransfer(TBAAAccessInfo DestInfo,
                                             TBAAAccessInfo SrcInfo) {
  if (!tbaa)
    return TBAAAccessInfo();
  return tbaa->mergeTBAAInfoForConditionalOperator(DestInfo, SrcInfo);
}
