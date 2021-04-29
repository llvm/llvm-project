// Test that we print pass structure with new and legacy PM.
// RUN: %clang -fexperimental-new-pass-manager -fdebug-pass-structure -O3 -S -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NEWPM
// RUN: %clang -flegacy-pass-manager -fdebug-pass-structure -O0 -S -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=LEGACYPM
// REQUIRES: asserts

// NEWPM: Annotation2MetadataPass on [module]
// NEWPM-NEXT: ForceFunctionAttrsPass on [module]
// NEWPM-NEXT: InferFunctionAttrsPass on [module]
// NEWPM-NEXT:   InnerAnalysisManagerProxy<{{.*}}> analysis on [module]
// NEWPM-NEXT: ModuleToFunctionPassAdaptor on [module]
// NEWPM-NEXT: OpenMPOptPass on [module]
// NEWPM-NEXT: IPSCCPPass on [module]
// NEWPM-NEXT: CalledValuePropagationPass on [module]
// NEWPM-NEXT: GlobalOptPass on [module]
// NEWPM-NEXT: ModuleToFunctionPassAdaptor on [module]
// NEWPM-NEXT: DeadArgumentEliminationPass on [module]
// NEWPM-NEXT: ModuleToFunctionPassAdaptor on [module]
// NEWPM-NEXT: ModuleInlinerWrapperPass on [module]
// NEWPM-NEXT:   InlineAdvisorAnalysis analysis on [module]
// NEWPM-NEXT:   RequireAnalysisPass<{{.*}}> on [module]
// NEWPM-NEXT:     GlobalsAA analysis on [module]
// NEWPM-NEXT:       CallGraphAnalysis analysis on [module]
// NEWPM-NEXT:   RequireAnalysisPass<{{.*}}> on [module]
// NEWPM-NEXT:     ProfileSummaryAnalysis analysis on [module]
// NEWPM-NEXT:   ModuleToPostOrderCGSCCPassAdaptor on [module]
// NEWPM-NEXT:     InnerAnalysisManagerProxy<{{.*}}> analysis on [module]
// NEWPM-NEXT:       LazyCallGraphAnalysis analysis on [module]
// NEWPM-NEXT: GlobalOptPass on [module]
// NEWPM-NEXT: GlobalDCEPass on [module]
// NEWPM-NEXT: EliminateAvailableExternallyPass on [module]
// NEWPM-NEXT: ReversePostOrderFunctionAttrsPass on [module]
// NEWPM-NEXT: RequireAnalysisPass<{{.*}}> on [module]
// NEWPM-NEXT: ModuleToFunctionPassAdaptor on [module]
// NEWPM-NEXT: CGProfilePass on [module]
// NEWPM-NEXT: GlobalDCEPass on [module]
// NEWPM-NEXT: ConstantMergePass on [module]
// NEWPM-NEXT: RelLookupTableConverterPass on [module]
// NEWPM-NEXT: ModuleToFunctionPassAdaptor on [module]
// NEWPM-NEXT: PrintModulePass on [module]

// LEGACYPM:      Pass Arguments:  -tti -targetlibinfo -verify
// LEGACYPM-NEXT: Target Transform Information
// LEGACYPM-NEXT: Target Library Information
// LEGACYPM-NEXT:   FunctionPass Manager
// LEGACYPM-NEXT:     Module Verifier
// LEGACYPM-NEXT: Pass Arguments:  -tti -targetlibinfo -assumption-cache-tracker -profile-summary-info -annotation2metadata -forceattrs -basiccg -always-inline -annotation-remarks
// LEGACYPM-NEXT: Target Transform Information
// LEGACYPM-NEXT: Target Library Information
// LEGACYPM-NEXT: Assumption Cache Tracker
// LEGACYPM-NEXT: Profile summary info
// LEGACYPM-NEXT:   ModulePass Manager
// LEGACYPM-NEXT:     Annotation2Metadata
// LEGACYPM-NEXT:     Force set function attributes
// LEGACYPM-NEXT:     CallGraph Construction
// LEGACYPM-NEXT:     Call Graph SCC Pass Manager
// LEGACYPM-NEXT:       Inliner for always_inline functions
// LEGACYPM-NEXT:       FunctionPass Manager
// LEGACYPM-NEXT:         Annotation Remarks
// LEGACYPM-NEXT:     Print Module IR
// LEGACYPM-NEXT: Pass Arguments:  -tti
// LEGACYPM-NEXT: Target Transform Information
// LEGACYPM-NEXT:   ModulePass Manager

