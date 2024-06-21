! This test checks lowering of OpenMP Threadprivate Directive.
! Test for threadprivate variable in host association.

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK: func.func @_QQmain() attributes {fir.bindc_name = "main"} {
!CHECK:   %[[A:.*]] = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFEa"}
!CHECK:   %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] {uniq_name = "_QFEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   %[[A_ADDR:.*]] = fir.address_of(@_QFEa) : !fir.ref<i32>
!CHECK:   %[[TP_A:.*]] = omp.threadprivate %[[A_ADDR]] : !fir.ref<i32> -> !fir.ref<i32>
!CHECK:   %[[TP_A_DECL:.*]]:2 = hlfir.declare %[[TP_A]] {uniq_name = "_QFEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   fir.call @_QFPsub() fastmath<contract> : () -> ()
!CHECK:   return
!CHECK: }
!CHECK: func.func private @_QFPsub() attributes {fir.host_symbol = {{.*}}, llvm.linkage = #llvm.linkage<internal>} {
!CHECK:   %[[A:.*]] = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFEa"}
!CHECK:   %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] {uniq_name = "_QFEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   %[[A_ADDR:.*]] = fir.address_of(@_QFEa) : !fir.ref<i32>
!CHECK:   %[[TP_A:.*]] = omp.threadprivate %[[A_ADDR]] : !fir.ref<i32> -> !fir.ref<i32>
!CHECK:   %[[TP_A_DECL:.*]]:2 = hlfir.declare %[[TP_A]] {uniq_name = "_QFEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   omp.parallel {
!CHECK:     %[[PAR_TP_A:.*]] = omp.threadprivate %[[A_ADDR]] : !fir.ref<i32> -> !fir.ref<i32>
!CHECK:     %[[PAR_TP_A_DECL:.*]]:2 = hlfir.declare %[[PAR_TP_A]] {uniq_name = "_QFEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:     %{{.*}} = fir.load %[[PAR_TP_A_DECL]]#0 : !fir.ref<i32>
!CHECK:     omp.terminator
!CHECK:   }
!CHECK:   return
!CHECK: }
!CHECK: fir.global internal @_QFEa : i32 {
!CHECK:   %[[A:.*]] = fir.undefined i32
!CHECK:   fir.has_value %[[A]] : i32
!CHECK: }

program main
   integer :: a
   !$omp threadprivate(a)
   call sub()
contains
   subroutine sub()
      !$omp parallel
      print *, a
      !$omp end parallel
   end
end
