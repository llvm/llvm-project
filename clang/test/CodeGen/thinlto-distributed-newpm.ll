; FIXME: This test should use CHECK-NEXT to keep up-to-date.
; REQUIRES: x86-registered-target

; Validate ThinLTO post link pipeline at O2 and O3

; RUN: opt -thinlto-bc -o %t.o %s

; RUN: llvm-lto2 run -thinlto-distributed-indexes %t.o \
; RUN:   -o %t2.index \
; RUN:   -r=%t.o,main,px

; RUN: %clang -target x86_64-grtev4-linux-gnu \
; RUN:   -O2 -fexperimental-new-pass-manager -Xclang -fdebug-pass-manager \
; RUN:   -c -fthinlto-index=%t.o.thinlto.bc \
; RUN:   -o %t.native.o -x ir %t.o 2>&1 | FileCheck -check-prefix=CHECK-O %s --dump-input=fail

; RUN: %clang -target x86_64-grtev4-linux-gnu \
; RUN:   -O3 -fexperimental-new-pass-manager -Xclang -fdebug-pass-manager \
; RUN:   -c -fthinlto-index=%t.o.thinlto.bc \
; RUN:   -o %t.native.o -x ir %t.o 2>&1 | FileCheck -check-prefixes=CHECK-O,CHECK-O3 %s --dump-input=fail

; CHECK-O: Starting {{.*}}Module pass manager run.
; CHECK-O: Running pass: WholeProgramDevirtPass
; CHECK-O: Running analysis: InnerAnalysisManagerProxy
; CHECK-O: Running pass: LowerTypeTestsPass
; CHECK-O: Invalidating analysis: InnerAnalysisManagerProxy
; CHECK-O: Running pass: ForceFunctionAttrsPass
; CHECK-O: Running pass: PGOIndirectCallPromotion
; CHECK-O: Running analysis: ProfileSummaryAnalysis
; CHECK-O: Running analysis: InnerAnalysisManagerProxy
; CHECK-O: Running analysis: OptimizationRemarkEmitterAnalysis on main
; CHECK-O: Running pass: InferFunctionAttrsPass
; CHECK-O: Starting {{.*}}Function pass manager run.
; CHECK-O: Running pass: SimplifyCFGPass on main
; CHECK-O: Running analysis: TargetIRAnalysis on main
; CHECK-O: Running analysis: AssumptionAnalysis on main
; CHECK-O: Running pass: SROA on main
; CHECK-O: Running analysis: DominatorTreeAnalysis on main
; CHECK-O: Running pass: EarlyCSEPass on main
; CHECK-O: Running analysis: TargetLibraryAnalysis on main
; CHECK-O: Running pass: LowerExpectIntrinsicPass on main
; CHECK-O3: Running pass: CallSiteSplittingPass on main
; CHECK-O: Finished {{.*}}Function pass manager run.
; CHECK-O: Running pass: LowerTypeTestsPass
; CHECK-O: Running pass: IPSCCPPass
; CHECK-O: Running pass: CalledValuePropagationPass
; CHECK-O: Running pass: GlobalOptPass
; CHECK-O: Invalidating analysis: InnerAnalysisManagerProxy
; CHECK-O: Running analysis: InnerAnalysisManagerProxy
; CHECK-O: Running pass: PromotePass
; CHECK-O: Running analysis: DominatorTreeAnalysis on main
; CHECK-O: Running analysis: AssumptionAnalysis on main
; CHECK-O: Running pass: DeadArgumentEliminationPass
; CHECK-O: Starting {{.*}}Function pass manager run.
; CHECK-O: Running pass: InstCombinePass on main
; CHECK-O: Running analysis: TargetLibraryAnalysis on main
; CHECK-O: Running analysis: OptimizationRemarkEmitterAnalysis on main
; CHECK-O: Running analysis: TargetIRAnalysis on main
; CHECK-O: Running analysis: AAManager on main
; CHECK-O: Running analysis: BasicAA on main
; CHECK-O: Running analysis: ScopedNoAliasAA on main
; CHECK-O: Running analysis: TypeBasedAA on main
; CHECK-O: Running analysis: OuterAnalysisManagerProxy
; CHECK-O: Running pass: SimplifyCFGPass on main
; CHECK-O: Finished {{.*}}Function pass manager run.
; CHECK-O: Running analysis: InnerAnalysisManagerProxy
; CHECK-O: Running analysis: LazyCallGraphAnalysis
; CHECK-O: Running analysis: FunctionAnalysisManagerCGSCCProxy on (main)
; CHECK-O: Running analysis: OuterAnalysisManagerProxy
; CHECK-O: Starting CGSCC pass manager run.
; CHECK-O: Running pass: InlinerPass on (main)
; CHECK-O: Running pass: PostOrderFunctionAttrsPass on (main)
; CHECK-O: Invalidating analysis: DominatorTreeAnalysis on main
; CHECK-O: Invalidating analysis: BasicAA on main
; CHECK-O: Invalidating analysis: AAManager on main
; CHECK-O3: Running pass: ArgumentPromotionPass on (main)
; CHECK-O: Starting {{.*}}Function pass manager run.
; CHECK-O: Running pass: SROA on main
; These next two can appear in any order since they are accessed as parameters
; on the same call to SROA::runImpl
; CHECK-O-DAG: Running analysis: DominatorTreeAnalysis on main
; CHECK-O: Running pass: EarlyCSEPass on main
; CHECK-O: Running analysis: MemorySSAAnalysis on main
; CHECK-O: Running analysis: AAManager on main
; CHECK-O: Running analysis: BasicAA on main
; CHECK-O: Running pass: SpeculativeExecutionPass on main
; CHECK-O: Running pass: JumpThreadingPass on main
; CHECK-O: Running analysis: LazyValueAnalysis on main
; CHECK-O: Running pass: CorrelatedValuePropagationPass on main
; CHECK-O: Invalidating analysis: LazyValueAnalysis on main
; CHECK-O: Running pass: SimplifyCFGPass on main
; CHECK-O3: Running pass: AggressiveInstCombinePass on main
; CHECK-O: Running pass: InstCombinePass on main
; CHECK-O: Running pass: LibCallsShrinkWrapPass on main
; CHECK-O: Running pass: TailCallElimPass on main
; CHECK-O: Running pass: SimplifyCFGPass on main
; CHECK-O: Running pass: ReassociatePass on main
; CHECK-O: Running pass: RequireAnalysisPass<{{.*}}OptimizationRemarkEmitterAnalysis
; CHECK-O: Starting {{.*}}Function pass manager run.
; CHECK-O: Running pass: LoopSimplifyPass on main
; CHECK-O: Running analysis: LoopAnalysis on main
; CHECK-O: Running pass: LCSSAPass on main
; CHECK-O: Finished {{.*}}Function pass manager run.
; CHECK-O: Running pass: SimplifyCFGPass on main
; CHECK-O: Running pass: InstCombinePass on main
; CHECK-O: Starting {{.*}}Function pass manager run.
; CHECK-O: Running pass: LoopSimplifyPass on main
; CHECK-O: Running pass: LCSSAPass on main
; CHECK-O: Finished {{.*}}Function pass manager run.
; CHECK-O: Running pass: SROA on main
; CHECK-O: Running pass: MergedLoadStoreMotionPass on main
; CHECK-O: Running pass: GVN on main
; CHECK-O: Running pass: SCCPPass on main
; CHECK-O: Running pass: BDCEPass on main
; CHECK-O: Running analysis: DemandedBitsAnalysis on main
; CHECK-O: Running pass: InstCombinePass on main
; CHECK-O: Running pass: JumpThreadingPass on main
; CHECK-O: Running pass: CorrelatedValuePropagationPass on main
; CHECK-O: Running pass: ADCEPass on main
; CHECK-O: Running analysis: PostDominatorTreeAnalysis on main
; CHECK-O: Running pass: MemCpyOptPass on main
; CHECK-O: Running pass: DSEPass on main
; CHECK-O: Starting {{.*}}Function pass manager run.
; CHECK-O: Running pass: LoopSimplifyPass on main
; CHECK-O: Running pass: LCSSAPass on main
; CHECK-O: Finished {{.*}}Function pass manager run.
; CHECK-O: Running pass: SimplifyCFGPass on main
; CHECK-O: Running pass: InstCombinePass on main
; CHECK-O: Finished {{.*}}Function pass manager run.
; CHECK-O: Finished CGSCC pass manager run.
; CHECK-O: Invalidating analysis: DominatorTreeAnalysis on main
; CHECK-O: Invalidating analysis: BasicAA on main
; CHECK-O: Invalidating analysis: AAManager on main
; CHECK-O: Invalidating analysis: MemorySSAAnalysis on main
; CHECK-O: Invalidating analysis: LoopAnalysis on main
; CHECK-O: Invalidating analysis: PhiValuesAnalysis on main
; CHECK-O: Invalidating analysis: MemoryDependenceAnalysis on main
; CHECK-O: Invalidating analysis: DemandedBitsAnalysis on main
; CHECK-O: Invalidating analysis: PostDominatorTreeAnalysis on main
; CHECK-O: Invalidating analysis: CallGraphAnalysis
; CHECK-O: Running pass: GlobalOptPass
; CHECK-O: Running pass: GlobalDCEPass
; CHECK-O: Running pass: EliminateAvailableExternallyPass
; CHECK-O: Running pass: ReversePostOrderFunctionAttrsPass
; CHECK-O: Running analysis: CallGraphAnalysis
; CHECK-O: Running pass: RequireAnalysisPass<{{.*}}GlobalsAA
; CHECK-O: Starting {{.*}}Function pass manager run.
; CHECK-O: Running pass: Float2IntPass on main
; CHECK-O: Running pass: LowerConstantIntrinsicsPass on main
; CHECK-O: Starting {{.*}}Function pass manager run.
; CHECK-O: Running pass: LoopSimplifyPass on main
; CHECK-O: Running analysis: LoopAnalysis on main
; CHECK-O: Running pass: LCSSAPass on main
; CHECK-O: Finished {{.*}}Function pass manager run.
; CHECK-O: Running analysis: MemorySSAAnalysis on main
; CHECK-O: Running analysis: AAManager on main
; CHECK-O: Running analysis: BasicAA on main
; CHECK-O: Running analysis: ScalarEvolutionAnalysis on main
; CHECK-O: Running analysis: InnerAnalysisManagerProxy
; CHECK-O: Running pass: LoopRotatePass on Loop at depth 1 containing: %b
; CHECK-O: Running pass: LoopDistributePass on main
; CHECK-O: Running pass: InjectTLIMappings on main
; CHECK-O: Running pass: LoopVectorizePass on main
; CHECK-O: Running analysis: BlockFrequencyAnalysis on main
; CHECK-O: Running analysis: BranchProbabilityAnalysis on main
; CHECK-O: Running analysis: PostDominatorTreeAnalysis on main
; CHECK-O: Running analysis: DemandedBitsAnalysis on main
; CHECK-O: Running pass: LoopLoadEliminationPass on main
; CHECK-O: Running pass: InstCombinePass on main
; CHECK-O: Running pass: SimplifyCFGPass on main
; CHECK-O: Running pass: SLPVectorizerPass on main
; CHECK-O: Running pass: VectorCombinePass on main
; CHECK-O: Running pass: InstCombinePass on main
; CHECK-O: Running pass: LoopUnrollPass on main
; CHECK-O: Running pass: WarnMissedTransformationsPass on main
; CHECK-O: Running pass: InstCombinePass on main
; CHECK-O: Running pass: RequireAnalysisPass<{{.*}}OptimizationRemarkEmitterAnalysis
; CHECK-O: Starting {{.*}}Function pass manager run.
; CHECK-O: Running pass: LoopSimplifyPass on main
; CHECK-O: Running pass: LCSSAPass on main
; CHECK-O: Finished {{.*}}Function pass manager run.
; CHECK-O: Running pass: LICMPass on Loop at depth 1 containing: %b
; CHECK-O: Running pass: AlignmentFromAssumptionsPass on main
; CHECK-O: Running pass: LoopSinkPass on main
; CHECK-O: Running pass: InstSimplifyPass on main
; CHECK-O: Running pass: DivRemPairsPass on main
; CHECK-O: Running pass: SimplifyCFGPass on main
; CHECK-O: Running pass: SpeculateAroundPHIsPass on main
; CHECK-O: Finished {{.*}}Function pass manager run.
; CHECK-O: Running pass: CGProfilePass
; CHECK-O: Running pass: GlobalDCEPass
; CHECK-O: Running pass: ConstantMergePass
; CHECK-O: Finished {{.*}}Module pass manager run.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

define i32 @main() {
  br label %b
b:
  br label %b
  ret i32 0
}
