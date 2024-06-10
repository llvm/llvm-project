! Tests that CFG & LLVM conversion is applied to `omp.private` ops.

! RUN: split-file %s %t && cd %t

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - test.f90 2>&1 | \
! RUN: fir-opt --cfg-conversion -o test.cfg-conv.mlir
! RUN: FileCheck --input-file=test.cfg-conv.mlir %s --check-prefix="CFGConv"

! RUN: fir-opt --convert-hlfir-to-fir --cg-rewrite --fir-to-llvm-ir test.cfg-conv.mlir -o - | \
! RUN: FileCheck %s --check-prefix="LLVMDialect"

!--- test.f90
subroutine delayed_privatization_allocatable
  implicit none
  integer, allocatable :: var1

!$omp parallel private(var1)
  var1 = 10
!$omp end parallel
end subroutine

! CFGConv-LABEL: omp.private {type = private}
! CFGConv-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.ref<!fir.box<!fir.heap<i32>>>]] alloc {

! CFGConv-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):

! CFGConv-NEXT:   %[[PRIV_ALLOC:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "var1", pinned, uniq_name = "_QFdelayed_privatization_allocatableEvar1"}

! CFGConv-NEXT:   %[[PRIV_ARG_VAL:.*]] = fir.load %[[PRIV_ARG]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CFGConv-NEXT:   %[[PRIV_ARG_BOX:.*]] = fir.box_addr %[[PRIV_ARG_VAL]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CFGConv-NEXT:   %[[PRIV_ARG_ADDR:.*]] = fir.convert %[[PRIV_ARG_BOX]] : (!fir.heap<i32>) -> i64
! CFGConv-NEXT:   %[[C0:.*]] = arith.constant 0 : i64
! CFGConv-NEXT:   %[[ALLOC_COND:.*]] = arith.cmpi ne, %[[PRIV_ARG_ADDR]], %[[C0]] : i64
! CFGConv-NEXT:   cf.cond_br %[[ALLOC_COND]], ^[[ALLOC_MEM_BB:.*]], ^[[ZERO_MEM_BB:.*]]
! CFGConv-NEXT: ^[[ALLOC_MEM_BB]]:
! CFGConv:        fir.allocmem
! CFGConv:        cf.br ^[[DECL_BB:.*]]
! CFGConv:      ^[[ZERO_MEM_BB]]:
! CFGConv-NEXT:   fir.zero_bits
! CFGConv:        cf.br ^[[DECL_BB:.*]]
! CFGConv-NEXT: ^[[DECL_BB]]:
! CFGConv-NEXT:   hlfir.declare
! CFGConv-NEXT:   omp.yield


! LLVMDialect-LABEL: omp.private {type = private}
! LLVMDialect-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!llvm.ptr]] alloc {

! LLVMDialect-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):
! LLVMDialect:        llvm.alloca
! LLVMDialect:        llvm.call @malloc

! LLVMDialect-NOT:    hlfir.declare
