! Test lowering of COPYPRIVATE with an unlimited polymorphic pointer.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! Testcase from: https://github.com/llvm/llvm-project/issues/198770

! CHECK-LABEL: func.func private @_copy_ref_class_ptr_none(
! CHECK-SAME:    %arg0: [[TYPE:!fir.ref<!fir.class<!fir.ptr<none>>>]],
! CHECK-SAME:    %arg1: [[TYPE]]) attributes {llvm.linkage = #llvm.linkage<internal>} {
! CHECK:   %[[DST:.*]]:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_copy_ref_class_ptr_none_dst"}
! CHECK:   %[[SRC:.*]]:2 = hlfir.declare %arg1 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_copy_ref_class_ptr_none_src"}
! CHECK:   %[[LD:.*]] = fir.load %[[SRC]]#0 : [[TYPE]]
! CHECK:   fir.store %[[LD]] to %[[DST]]#0 : [[TYPE]]
! CHECK:   return
! CHECK: }

! CHECK-LABEL: func.func @_QQmain()
! CHECK: omp.single copyprivate(%{{.*}} -> @_copy_ref_class_ptr_none : !fir.ref<!fir.class<!fir.ptr<none>>>) {
! CHECK:   omp.terminator
! CHECK: }

class(*), pointer, save :: aa
!$omp threadprivate(aa)
!$omp single
!$omp end single copyprivate(aa)
end

