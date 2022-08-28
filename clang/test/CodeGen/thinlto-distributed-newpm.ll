; FIXME: This test should use CHECK-NEXT to keep up-to-date.
; REQUIRES: x86-registered-target

; Validate ThinLTO post link pipeline at O2 and O3

; RUN: opt -thinlto-bc -o %t.o %s

; RUN: llvm-lto2 run -thinlto-distributed-indexes %t.o \
; RUN:   -o %t2.index \
; RUN:   -r=%t.o,main,px

; RUN: %clang -target x86_64-grtev4-linux-gnu \
; RUN:   -O2 -Xclang -fdebug-pass-manager \
; RUN:   -c -fthinlto-index=%t.o.thinlto.bc \
; RUN:   -o %t.native.o -x ir %t.o 2>&1 | FileCheck -check-prefix=CHECK-O %s --dump-input=fail

; RUN: %clang -target x86_64-grtev4-linux-gnu \
; RUN:   -O3 -Xclang -fdebug-pass-manager \
; RUN:   -c -fthinlto-index=%t.o.thinlto.bc \
; RUN:   -o %t.native.o -x ir %t.o 2>&1 | FileCheck -check-prefixes=CHECK-O,CHECK-O3 %s --dump-input=fail

; CHECK-O: Running pass: WholeProgramDevirtPass
; CHECK-O: Running pass: LowerTypeTestsPass
; CHECK-O: Running pass: ForceFunctionAttrsPass
; CHECK-O: Running pass: PGOIndirectCallPromotion
; CHECK-O: Running pass: InferFunctionAttrsPass
; CHECK-O: Running pass: LowerExpectIntrinsicPass on main
; CHECK-O: Running pass: SimplifyCFGPass on main
; CHECK-O: Running pass: SROAPass on main
; CHECK-O: Running pass: EarlyCSEPass on main
; CHECK-O3: Running pass: CallSiteSplittingPass on main
; CHECK-O: Running pass: LowerTypeTestsPass
; CHECK-O: Running pass: IPSCCPPass
; CHECK-O: Running pass: CalledValuePropagationPass
; CHECK-O: Running pass: GlobalOptPass
; CHECK-O: Running pass: PromotePass
; CHECK-O: Running pass: InstCombinePass on main
; CHECK-O: Running pass: SimplifyCFGPass on main
; CHECK-O: Running pass: InlinerPass on (main)
; CHECK-O: Running pass: PostOrderFunctionAttrsPass on (main)
; CHECK-O3: Running pass: ArgumentPromotionPass on (main)
; CHECK-O: Running pass: SROAPass on main
; CHECK-O: Running pass: EarlyCSEPass on main
; CHECK-O: Running pass: SpeculativeExecutionPass on main
; CHECK-O: Running pass: JumpThreadingPass on main
; CHECK-O: Running pass: CorrelatedValuePropagationPass on main
; CHECK-O: Running pass: SimplifyCFGPass on main
; CHECK-O: Running pass: InstCombinePass on main
; CHECK-O3: Running pass: AggressiveInstCombinePass on main
; CHECK-O: Running pass: LibCallsShrinkWrapPass on main
; CHECK-O: Running pass: TailCallElimPass on main
; CHECK-O: Running pass: SimplifyCFGPass on main
; CHECK-O: Running pass: ReassociatePass on main
; CHECK-O: Running pass: RequireAnalysisPass<{{.*}}OptimizationRemarkEmitterAnalysis
; CHECK-O: Running pass: LoopSimplifyPass on main
; CHECK-O: Running pass: LCSSAPass on main
; CHECK-O: Running pass: SimplifyCFGPass on main
; CHECK-O: Running pass: InstCombinePass on main
; CHECK-O: Running pass: LoopSimplifyPass on main
; CHECK-O: Running pass: LCSSAPass on main
; CHECK-O: Running pass: SROAPass on main
; CHECK-O: Running pass: MergedLoadStoreMotionPass on main
; CHECK-O: Running pass: GVNPass on main
; CHECK-O: Running pass: SCCPPass on main
; CHECK-O: Running pass: BDCEPass on main
; CHECK-O: Running pass: InstCombinePass on main
; CHECK-O: Running pass: JumpThreadingPass on main
; CHECK-O: Running pass: CorrelatedValuePropagationPass on main
; CHECK-O: Running pass: ADCEPass on main
; CHECK-O: Running pass: MemCpyOptPass on main
; CHECK-O: Running pass: DSEPass on main
; CHECK-O: Running pass: LoopSimplifyPass on main
; CHECK-O: Running pass: LCSSAPass on main
; CHECK-O: Running pass: SimplifyCFGPass on main
; CHECK-O: Running pass: InstCombinePass on main
; CHECK-O: Running pass: DeadArgumentEliminationPass
; CHECK-O: Running pass: GlobalOptPass
; CHECK-O: Running pass: GlobalDCEPass
; CHECK-O: Running pass: EliminateAvailableExternallyPass
; CHECK-O: Running pass: ReversePostOrderFunctionAttrsPass
; CHECK-O: Running pass: RecomputeGlobalsAAPass
; CHECK-O: Running pass: Float2IntPass on main
; CHECK-O: Running pass: LowerConstantIntrinsicsPass on main
; CHECK-O: Running pass: LoopSimplifyPass on main
; CHECK-O: Running pass: LCSSAPass on main
; CHECK-O: Running pass: LoopRotatePass on Loop at depth 1 containing: %b
; CHECK-O: Running pass: LoopDistributePass on main
; CHECK-O: Running pass: InjectTLIMappings on main
; CHECK-O: Running pass: LoopVectorizePass on main
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
; CHECK-O: Running pass: LoopSimplifyPass on main
; CHECK-O: Running pass: LCSSAPass on main
; CHECK-O: Running pass: LICMPass on Loop at depth 1 containing: %b
; CHECK-O: Running pass: AlignmentFromAssumptionsPass on main
; CHECK-O: Running pass: LoopSinkPass on main
; CHECK-O: Running pass: InstSimplifyPass on main
; CHECK-O: Running pass: DivRemPairsPass on main
; CHECK-O: Running pass: SimplifyCFGPass on main
; CHECK-O: Running pass: GlobalDCEPass
; CHECK-O: Running pass: ConstantMergePass
; CHECK-O: Running pass: CGProfilePass

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

define i32 @main() {
  br label %b
b:
  br label %b
  ret i32 0
}
