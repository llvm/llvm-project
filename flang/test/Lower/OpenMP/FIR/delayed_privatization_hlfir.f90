! TODO Convert this file into a bunch of lit tests for each conversion step.

! RUN: bbc -fopenmp -emit-hlfir --openmp-enable-delayed-privatization %s -o - 

subroutine delayed_privatization()
  implicit none
  integer :: var1
  integer :: var2

  var1 = 111
  var2 = 222

!$OMP PARALLEL FIRSTPRIVATE(var1, var2)
  var1 = var1 + var2 + 2
!$OMP END PARALLEL

end subroutine


! -----------------------------------------
! ## This is what flang emits with the PoC:
! -----------------------------------------
!
! ----------------------------
! ### Conversion to HLFIR + OMP:
! ----------------------------
!module {
!  func.func @_QPdelayed_privatization() {
!    %0 = fir.alloca i32 {bindc_name = "var1", uniq_name = "_QFdelayed_privatizationEvar1"}
!    %1:2 = hlfir.declare %0 {uniq_name = "_QFdelayed_privatizationEvar1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!    %2 = fir.alloca i32 {bindc_name = "var2", uniq_name = "_QFdelayed_privatizationEvar2"}
!    %3:2 = hlfir.declare %2 {uniq_name = "_QFdelayed_privatizationEvar2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!    %c111_i32 = arith.constant 111 : i32
!    hlfir.assign %c111_i32 to %1#0 : i32, !fir.ref<i32>
!    %c222_i32 = arith.constant 222 : i32
!    hlfir.assign %c222_i32 to %3#0 : i32, !fir.ref<i32>
!    omp.parallel private(@var1.privatizer_0 %1#0, @var2.privatizer_0 %3#0 : !fir.ref<i32>, !fir.ref<i32>) {
!    ^bb0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>):
!      %4:2 = hlfir.declare %arg0 {uniq_name = "_QFdelayed_privatizationEvar1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!      %5:2 = hlfir.declare %arg1 {uniq_name = "_QFdelayed_privatizationEvar2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!      %6 = fir.load %4#0 : !fir.ref<i32>
!      %7 = fir.load %5#0 : !fir.ref<i32>
!      %8 = arith.addi %6, %7 : i32
!      %c2_i32 = arith.constant 2 : i32
!      %9 = arith.addi %8, %c2_i32 : i32
!      hlfir.assign %9 to %4#0 : i32, !fir.ref<i32>
!      omp.terminator
!    }
!    return
!  }
!  "omp.private"() <{function_type = (!fir.ref<i32>) -> !fir.ref<i32>, sym_name = "var1.privatizer_0"}> ({
!  ^bb0(%arg0: !fir.ref<i32>):
!    %0 = fir.alloca i32 {bindc_name = "var1", pinned, uniq_name = "_QFdelayed_privatizationEvar1"}
!    %1:2 = hlfir.declare %0 {uniq_name = "_QFdelayed_privatizationEvar1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!    %2 = fir.load %arg0 : !fir.ref<i32>
!    hlfir.assign %2 to %1#0 temporary_lhs : i32, !fir.ref<i32>
!    omp.yield(%1#0 : !fir.ref<i32>)
!  }) : () -> ()
!  "omp.private"() <{function_type = (!fir.ref<i32>) -> !fir.ref<i32>, sym_name = "var2.privatizer_0"}> ({
!  ^bb0(%arg0: !fir.ref<i32>):
!    %0 = fir.alloca i32 {bindc_name = "var2", pinned, uniq_name = "_QFdelayed_privatizationEvar2"}
!    %1:2 = hlfir.declare %0 {uniq_name = "_QFdelayed_privatizationEvar2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!    %2 = fir.load %arg0 : !fir.ref<i32>
!    hlfir.assign %2 to %1#0 temporary_lhs : i32, !fir.ref<i32>
!    omp.yield(%1#0 : !fir.ref<i32>)
!  }) : () -> ()
!}
!
!
! ### After lowring `hlfir` to `fir`, conversion to LLVM + OMP -> LLVM IR produces the exact same result as for
! `delayed_privatization.f90`.
