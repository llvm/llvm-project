//===---- MissingFeatures.h - Checks for unimplemented features -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file introduces some helper classes to guard against features that
// CIR dialect supports that we do not have and also do not have great ways to
// assert against.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_MISSINGFEATURES_H
#define CLANG_CIR_MISSINGFEATURES_H

#include <llvm/Support/raw_ostream.h>

constexpr bool cirCConvAssertionMode =
    true; // Change to `false` to use llvm_unreachable

#define CIR_CCONV_NOTE                                                         \
  " Target lowering is now required. To workaround use "                       \
  "-fno-clangir-call-conv-lowering. This flag is going to be removed at some"  \
  " point."

// Special assertion to be used in the target lowering library.
#define cir_cconv_assert(cond)                                                 \
  do {                                                                         \
    if (!(cond))                                                               \
      llvm::errs() << CIR_CCONV_NOTE << "\n";                                  \
    assert((cond));                                                            \
  } while (0)

// Special version of cir_cconv_unreachable to give more info to the user on how
// to temporaruly disable target lowering.
#define cir_cconv_unreachable(msg)                                             \
  do {                                                                         \
    llvm_unreachable(msg CIR_CCONV_NOTE);                                      \
  } while (0)

// Some assertions knowingly generate incorrect code. This macro allows us to
// switch between using `assert` and `llvm_unreachable` for these cases.
#define cir_cconv_assert_or_abort(cond, msg)                                   \
  do {                                                                         \
    if (cirCConvAssertionMode) {                                               \
      assert((cond) && msg CIR_CCONV_NOTE);                                    \
    } else {                                                                   \
      llvm_unreachable(msg CIR_CCONV_NOTE);                                    \
    }                                                                          \
  } while (0)

namespace cir {

struct MissingFeatures {
  // TODO(CIR): Implement the CIRGenFunction::emitTypeCheck method that handles
  // sanitizer related type check features
  static bool emitTypeCheck() { return false; }
  static bool tbaa() { return false; }
  static bool tbaaStruct() { return false; }
  static bool tbaaTagForStruct() { return false; }
  static bool tbaaTagForEnum() { return false; }
  static bool tbaaTagForBitInt() { return false; }
  static bool tbaaVTablePtr() { return false; }
  static bool tbaaIncompleteType() { return false; }
  static bool tbaaMergeTBAAInfo() { return false; }
  static bool tbaaMayAlias() { return false; }
  static bool tbaaNewStructPath() { return false; }
  static bool tbaaPointer() { return false; }
  static bool emitNullabilityCheck() { return false; }
  static bool ptrAuth() { return false; }
  static bool memberFuncPtrAuthInfo() { return false; }
  static bool emitCFICheck() { return false; }
  static bool emitVFEInfo() { return false; }
  static bool emitWPDInfo() { return false; }

  // GNU vectors are done, but other kinds of vectors haven't been implemented.
  static bool scalableVectors() { return false; }
  static bool vectorConstants() { return false; }

  // Address space related
  static bool addressSpace() { return false; }

  // Clang codegen options
  static bool strictVTablePointers() { return false; }

  // Unhandled global/linkage information.
  static bool unnamedAddr() { return false; }
  static bool setComdat() { return false; }
  static bool setDSOLocal() { return false; }
  static bool threadLocal() { return false; }
  static bool setDLLStorageClass() { return false; }
  static bool setDLLImportDLLExport() { return false; }
  static bool setPartition() { return false; }
  static bool setGlobalVisibility() { return false; }
  static bool hiddenVisibility() { return false; }
  static bool protectedVisibility() { return false; }
  static bool addCompilerUsedGlobal() { return false; }
  static bool supportIFuncAttr() { return false; }
  static bool setDefaultVisibility() { return false; }
  static bool addUsedOrCompilerUsedGlobal() { return false; }
  static bool addUsedGlobal() { return false; }
  static bool addSectionAttributes() { return false; }
  static bool setSectionForFuncOp() { return false; }
  static bool updateCPUAndFeaturesAttributes() { return false; }

  // Sanitizers
  static bool reportGlobalToASan() { return false; }
  static bool emitAsanPrologueOrEpilogue() { return false; }
  static bool emitCheckedInBoundsGEP() { return false; }
  static bool pointerOverflowSanitizer() { return false; }
  static bool sanitizeDtor() { return false; }
  static bool sanitizeVLABound() { return false; }
  static bool sanitizerBuiltin() { return false; }
  static bool sanitizerReturn() { return false; }
  static bool sanitizeOther() { return false; }

  // ObjC
  static bool setObjCGCLValueClass() { return false; }
  static bool objCLifetime() { return false; }
  static bool objCIvarDecls() { return false; }
  static bool objCRuntime() { return false; }

  // Debug info
  static bool generateDebugInfo() { return false; }
  static bool noDebugInfo() { return false; }

  // LLVM Attributes
  static bool setFunctionAttributes() { return false; }
  static bool attributeBuiltin() { return false; }
  static bool attributeNoBuiltin() { return false; }
  static bool parameterAttributes() { return false; }
  static bool minLegalVectorWidthAttr() { return false; }
  static bool vscaleRangeAttr() { return false; }
  static bool stackrealign() { return false; }
  static bool zerocallusedregs() { return false; }

  // Coroutines
  static bool unhandledException() { return false; }

  // Missing Emissions
  static bool variablyModifiedTypeEmission() { return false; }
  static bool emitLValueAlignmentAssumption() { return false; }
  static bool emitDerivedToBaseCastForDevirt() { return false; }
  static bool emitFunctionEpilog() { return false; }

  // References related stuff
  static bool ARC() { return false; } // Automatic reference counting

  // Clang early optimizations or things defered to LLVM lowering.
  static bool shouldUseBZeroPlusStoresToInitialize() { return false; }
  static bool shouldUseMemSetToInitialize() { return false; }
  static bool shouldSplitConstantStore() { return false; }
  static bool shouldCreateMemCpyFromGlobal() { return false; }
  static bool shouldReverseUnaryCondOnBoolExpr() { return false; }
  static bool isTrivialCtorOrDtor() { return false; }
  static bool isMemcpyEquivalentSpecialMember() { return false; }
  static bool constructABIArgDirectExtend() { return false; }
  static bool mayHaveIntegerOverflow() { return false; }
  static bool llvmLoweringPtrDiffConsidersPointee() { return false; }
  static bool emitNullCheckForDeleteCalls() { return false; }

  // Folding methods.
  static bool foldBinOpFMF() { return false; }

  // Fast math.
  static bool fastMathGuard() { return false; }
  // Should be implemented with a moduleOp level attribute and directly
  // mapped to LLVM - those can be set directly for every relevant LLVM IR
  // dialect operation (log10, ...).
  static bool fastMathFlags() { return false; }
  static bool fastMathFuncAttributes() { return false; }

  // Cleanup
  static bool cleanups() { return false; }
  static bool simplifyCleanupEntry() { return false; }
  static bool requiresCleanups() { return false; }
  static bool cleanupBranchAfterSwitch() { return false; }
  static bool cleanupAlwaysBranchThrough() { return false; }
  static bool cleanupDestinationIndex() { return false; }
  static bool cleanupDestroyNRVOVariable() { return false; }
  static bool cleanupAppendInsts() { return false; }
  static bool cleanupIndexAndBIAdjustment() { return false; }

  // Exception handling
  static bool isSEHTryScope() { return false; }
  static bool ehStack() { return false; }
  static bool emitStartEHSpec() { return false; }
  static bool emitEndEHSpec() { return false; }

  // Type qualifiers.
  static bool atomicTypes() { return false; }
  static bool volatileTypes() { return false; }
  static bool syncScopeID() { return false; }

  // ABIInfo queries.
  static bool useTargetLoweringABIInfo() { return false; }
  static bool isEmptyFieldForLayout() { return false; }

  // Misc
  static bool cacheRecordLayouts() { return false; }
  static bool capturedByInit() { return false; }
  static bool tryEmitAsConstant() { return false; }
  static bool incrementProfileCounter() { return false; }
  static bool createProfileWeightsForLoop() { return false; }
  static bool getProfileCount() { return false; }
  static bool emitCondLikelihoodViaExpectIntrinsic() { return false; }
  static bool requiresReturnValueCheck() { return false; }
  static bool shouldEmitLifetimeMarkers() { return false; }
  static bool peepholeProtection() { return false; }
  static bool CGCapturedStmtInfo() { return false; }
  static bool CGFPOptionsRAII() { return false; }
  static bool getFPFeaturesInEffect() { return false; }
  static bool cxxABI() { return false; }
  static bool openCLCXX() { return false; }
  static bool openCLBuiltinTypes() { return false; }
  static bool CUDA() { return false; }
  static bool openMP() { return false; }
  static bool openMPRuntime() { return false; }
  static bool openMPRegionInfo() { return false; }
  static bool openMPTarget() { return false; }
  static bool isVarArg() { return false; }
  static bool setNonGC() { return false; }
  static bool volatileLoadOrStore() { return false; }
  static bool armComputeVolatileBitfields() { return false; }
  static bool insertBuiltinUnpredictable() { return false; }
  static bool createInvariantGroup() { return false; }
  static bool createInvariantIntrinsic() { return false; }
  static bool addAutoInitAnnotation() { return false; }
  static bool addHeapAllocSiteMetadata() { return false; }
  static bool loopInfoStack() { return false; }
  static bool constantFoldsToSimpleInteger() { return false; }
  static bool checkFunctionCallABI() { return false; }
  static bool zeroInitializer() { return false; }
  static bool targetCodeGenInfoIsProtoCallVariadic() { return false; }
  static bool targetCodeGenInfoGetNullPointer() { return false; }
  static bool operandBundles() { return false; }
  static bool exceptions() { return false; }
  static bool metaDataNode() { return false; }
  static bool emitDeclMetadata() { return false; }
  static bool emitScalarRangeCheck() { return false; }
  static bool stmtExprEvaluation() { return false; }
  static bool setCallingConv() { return false; }
  static bool tryMarkNoThrow() { return false; }
  static bool indirectBranch() { return false; }
  static bool escapedLocals() { return false; }
  static bool deferredReplacements() { return false; }
  static bool shouldInstrumentFunction() { return false; }
  static bool xray() { return false; }
  static bool emitConstrainedFPCall() { return false; }
  static bool emitEmptyRecordCheck() { return false; }
  static bool isPPC_FP128Ty() { return false; }
  static bool createLaunderInvariantGroup() { return false; }

  // Inline assembly
  static bool asmGoto() { return false; }
  static bool asmUnwindClobber() { return false; }
  static bool asmMemoryEffects() { return false; }
  static bool asmVectorType() { return false; }
  static bool asmLLVMAssume() { return false; }

  // C++ ABI support
  static bool handleBigEndian() { return false; }
  static bool handleAArch64Indirect() { return false; }
  static bool classifyArgumentTypeForAArch64() { return false; }
  static bool supportgetCoerceToTypeForAArch64() { return false; }
  static bool supportTySizeQueryForAArch64() { return false; }
  static bool supportTyAlignQueryForAArch64() { return false; }
  static bool supportisHomogeneousAggregateQueryForAArch64() { return false; }
  static bool supportisEndianQueryForAArch64() { return false; }
  static bool supportisAggregateTypeForABIAArch64() { return false; }

  //===--- ABI lowering --===//

  static bool SPIRVABI() { return false; }

  static bool AArch64TypeClassification() { return false; }

  static bool X86ArgTypeClassification() { return false; }
  static bool X86DefaultABITypeConvertion() { return false; }
  static bool X86GetFPTypeAtOffset() { return false; }
  static bool X86RetTypeClassification() { return false; }
  static bool X86TypeClassification() { return false; }

  static bool ABIClangTypeKind() { return false; }
  static bool ABIFuncPtr() { return false; }
  static bool ABIInRegAttribute() { return false; }
  static bool ABINestedRecordLayout() { return false; }
  static bool ABINoProtoFunctions() { return false; }
  static bool ABIParameterCoercion() { return false; }
  static bool ABIPointerParameterAttrs() { return false; }
  static bool ABITransparentUnionHandling() { return false; }
  static bool ABIPotentialArgAccess() { return false; }
  static bool ABIByValAttribute() { return false; }
  static bool ABIAlignmentAttribute() { return false; }
  static bool ABINoAliasAttribute() { return false; }

  //-- Missing AST queries

  static bool CXXRecordDeclIsEmptyCXX11() { return false; }
  static bool CXXRecordDeclIsPOD() { return false; }
  static bool CXXRecordIsDynamicClass() { return false; }
  static bool astContextGetExternalSource() { return false; }
  static bool declGetMaxAlignment() { return false; }
  static bool declHasAlignMac68kAttr() { return false; }
  static bool declHasAlignNaturalAttr() { return false; }
  static bool declHasMaxFieldAlignmentAttr() { return false; }
  static bool fieldDeclIsBitfield() { return false; }
  static bool fieldDeclIsPotentiallyOverlapping() { return false; }
  static bool fieldDeclGetMaxFieldAlignment() { return false; }
  static bool fieldDeclisUnnamedBitField() { return false; }
  static bool funcDeclIsCXXConstructorDecl() { return false; }
  static bool funcDeclIsCXXDestructorDecl() { return false; }
  static bool funcDeclIsCXXMethodDecl() { return false; }
  static bool funcDeclIsInlineBuiltinDeclaration() { return false; }
  static bool funcDeclIsReplaceableGlobalAllocationFunction() { return false; }
  static bool isCXXRecordDecl() { return false; }
  static bool qualTypeIsReferenceType() { return false; }
  static bool recordDeclCanPassInRegisters() { return false; }
  static bool recordDeclHasAlignmentAttr() { return false; }
  static bool recordDeclHasFlexibleArrayMember() { return false; }
  static bool recordDeclIsCXXDecl() { return false; }
  static bool recordDeclIsMSStruct() { return false; }
  static bool recordDeclIsPacked() { return false; }
  static bool recordDeclMayInsertExtraPadding() { return false; }
  static bool typeGetAsBuiltinType() { return false; }
  static bool typeGetAsEnumType() { return false; }
  static bool typeIsCXXRecordDecl() { return false; }
  static bool typeIsScalableType() { return false; }
  static bool typeIsSized() { return false; }
  static bool varDeclIsKNRPromoted() { return false; }

  // We need to track parent (base) classes to determine the layout of a class.
  static bool getCXXRecordBases() { return false; }

  //-- Missing types

  static bool fixedWidthIntegers() { return false; }
  static bool vectorType() { return false; }
  static bool functionMemberPointerType() { return false; }
  static bool fixedSizeIntType() { return false; }

  //-- Missing LLVM attributes

  static bool noReturn() { return false; }
  static bool csmeCall() { return false; }
  static bool undef() { return false; }
  static bool noFPClass() { return false; }
  static bool llvmIntrinsicElementTypeSupport() { return false; }
  static bool argHasMaybeUndefAttr() { return false; }

  //-- Missing parts of the CIRGenModule::Release skeleton.
  static bool emitModuleInitializers() { return false; }
  static bool emittedDeferredDecls() { return false; }
  static bool emitVTablesOpportunistically() { return false; }
  static bool applyGlobalValReplacements() { return false; }
  static bool emitMultiVersionFunctions() { return false; }
  static bool incrementalExtensions() { return false; }
  static bool emitCXXModuleInitFunc() { return false; }
  static bool emitCXXGlobalCleanUpFunc() { return false; }
  static bool registerGlobalDtorsWithAtExit() { return false; }
  static bool emitCXXThreadLocalInitFunc() { return false; }
  static bool pgoReader() { return false; }
  static bool emitCtorList() { return false; }
  static bool emitStaticExternCAliases() { return false; }
  static bool checkAliases() { return false; }
  static bool emitDeferredUnusedCoverageMappings() { return false; }
  static bool cirGenPGO() { return false; }
  static bool coverageMapping() { return false; }
  static bool emitAtAvailableLinkGuard() { return false; }
  static bool emitLLVMUsed() { return false; }
  static bool sanStats() { return false; }
  static bool linkerOptionsMetadata() { return false; }
  static bool emitModuleLinkOptions() { return false; }
  static bool elfDependentLibraries() { return false; }
  static bool dwarfVersion() { return false; }
  static bool wcharWidth() { return false; }
  static bool enumWidth() { return false; }
  static bool setPICLevel() { return false; }
  static bool setPIELevel() { return false; }
  static bool codeModel() { return false; }
  static bool largeDataThreshold() { return false; }
  static bool directAccessExternalData() { return false; }
  static bool setFramePointer() { return false; }
  static bool simplifyPersonality() { return false; }
  static bool emitVersionIdentMetadata() { return false; }
  static bool emitTargetGlobals() { return false; }
  static bool emitTargetMetadata() { return false; }
  static bool emitBackendOptionsMetadata() { return false; }
  static bool embedObject() { return false; }
  static bool setVisibilityFromDLLStorageClass() { return false; }
  static bool mustTailCallUndefinedGlobals() { return false; }

  //-- Missing parts of the setCIRFunctionAttributesForDefinition skeleton.
  static bool stackProtector() { return false; }
  static bool optimizeForSize() { return false; }
  static bool minSize() { return false; }
  static bool setFunctionAlignment() { return false; }
  static bool memberFunctionPointerTypeMetadata() { return false; }

  //-- Other missing features

  // We need to track the parent record types that represent a field
  // declaration. This is necessary to determine the layout of a class.
  static bool fieldDeclAbstraction() { return false; }

  // There are some padding diagnostic features for Itanium ABI that we might
  // wanna add later.
  static bool bitFieldPaddingDiagnostics() { return false; }

  // Clang considers both enums and records as tag types. We don't have a way to
  // transparently handle both these types yet. Might need an interface here.
  static bool tagTypeClassAbstraction() { return false; }

  // Empty values might be passed as arguments to serve as padding, ensuring
  // alignment and compliance (e.g. MIPS). We do not yet support this.
  static bool argumentPadding() { return false; }

  // Clang has evaluation kinds which determines how code is emitted for certain
  // group of type classes. We don't have a way to identify type classes.
  static bool evaluationKind() { return false; }

  // Calls with a static chain pointer argument may be optimized (p.e. freeing
  // up argument registers), but we do not yet track such cases.
  static bool chainCall() { return false; }

  // ARM-specific feature that can be specified as a function attribute in C.
  static bool cmseNonSecureCallAttr() { return false; }

  // ABI-lowering has special handling for regcall calling convention (tries to
  // pass every argument in regs). We don't support it just yet.
  static bool regCall() { return false; }

  // Some ABIs (e.g. x86) require special handling for returning large structs
  // by value. The sret argument parameter aids in this, but it is current NYI.
  static bool sretArgs() { return false; }

  // Inalloca parameter attributes are mostly used for Windows x86_32 ABI. We
  // do not yet support this yet.
  static bool inallocaArgs() { return false; }

  // Parameters may have additional attributes (e.g. [[noescape]]) that affect
  // the compiler. This is not yet supported in CIR.
  static bool extParamInfo() { return false; }

  // LangOpts may affect lowering, but we do not carry this information into CIR
  // just yet. Right now, it only instantiates the default lang options.
  static bool langOpts() { return false; }

  // CodeGenOpts may affect lowering, but we do not carry this information into
  // CIR just yet. Right now, it only instantiates the default code generation
  // options.
  static bool codeGenOpts() { return false; }

  // Several type qualifiers are not yet supported in CIR, but important when
  // evaluating ABI-specific lowering.
  static bool qualifiedTypes() { return false; }

  // We're ignoring several details regarding ABI-handling for Swift.
  static bool swift() { return false; }

  // The AppleARM64 is using ItaniumCXXABI, which is not quite right.
  static bool appleArm64CXXABI() { return false; }

  // Despite carrying some information about variadics, we are currently
  // ignoring this to focus only on the code necessary to lower non-variadics.
  static bool variadicFunctions() { return false; }

  // If a store op is guaranteed to execute before the retun value load op, we
  // can optimize away the store and load ops. Seems like an early optimization.
  static bool returnValueDominatingStoreOptmiization() { return false; }

  // Globals (vars and functions) may have attributes that are target depedent.
  static bool setTargetAttributes() { return false; }

  // CIR modules parsed from text form may not carry the triple or data layout
  // specs. We should make it always present.
  static bool makeTripleAlwaysPresent() { return false; }

  static bool mustProgress() { return false; }

  static bool skipTempCopy() { return false; }
};

} // namespace cir

#endif // CLANG_CIR_MISSINGFEATURES_H
