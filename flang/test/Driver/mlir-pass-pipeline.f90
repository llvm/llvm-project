! Test the MLIR pass pipeline

! RUN: %flang_fc1 -S -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline %s -o - 2>&1 | FileCheck %s
end program

! CHECK: Pass statistics report

! CHECK: CSE
! CHECK-LABEL: 'func.func' Pipeline
! CHECK: ArrayValueCopy
! CHECK: CharacterConversion
! CHECK: Canonicalizer
! CHECK: SimplifyRegionLite
! CHECK: CSE

! CHECK-LABEL: 'func.func' Pipeline
! CHECK: MemoryAllocationOpt
! CHECK: Inliner
! CHECK: CSE

! CHECK-LABEL: 'func.func' Pipeline
! CHECK: CFGConversion
! CHECK: SCFToControlFlow
! CHECK: Canonicalizer
! CHECK: SimplifyRegionLite
! CHECK: CSE
! CHECK: BoxedProcedurePass

! CHECK-LABEL: 'func.func' Pipeline
! CHECK: AbstractResultOpt
! CHECK: CodeGenRewrite
! CHECK: TargetRewrite
! CHECK: ExternalNameConversion
! CHECK: FIRToLLVMLowering
! CHECK-NOT: LLVMIRLoweringPass
