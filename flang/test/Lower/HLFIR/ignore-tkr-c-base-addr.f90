! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

! Test that ignore_tkr(c) with a non-descriptor dummy (assumed-size) extracts
! the base address from allocatable/pointer actual arguments instead of passing
! the descriptor. This pattern is used by CUDA library interfaces like cuFFT.

module m_ignore_tkr_c_base_addr
  interface
    subroutine pass_assumed_size(a)
      !dir$ ignore_tkr(c) a
      real :: a(*)
    end subroutine
  end interface
contains
  ! CHECK-LABEL: func.func @_QMm_ignore_tkr_c_base_addrPtest_allocatable(
  ! CHECK-SAME: %[[ARR:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  subroutine test_allocatable(arr)
    real, allocatable :: arr(:)
    ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARR]]
    ! CHECK: %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
    ! CHECK: %[[CONV:.*]] = fir.convert %[[ADDR]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
    ! CHECK: fir.call @_QPpass_assumed_size(%[[CONV]]) {{.*}} : (!fir.ref<!fir.array<?xf32>>) -> ()
    call pass_assumed_size(arr)
  end subroutine

  ! CHECK-LABEL: func.func @_QMm_ignore_tkr_c_base_addrPtest_pointer(
  ! CHECK-SAME: %[[ARR:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  subroutine test_pointer(arr)
    real, pointer :: arr(:)
    ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARR]]
    ! CHECK: %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
    ! CHECK: %[[CONV:.*]] = fir.convert %[[ADDR]] : (!fir.ptr<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
    ! CHECK: fir.call @_QPpass_assumed_size(%[[CONV]]) {{.*}} : (!fir.ref<!fir.array<?xf32>>) -> ()
    call pass_assumed_size(arr)
  end subroutine

  ! CHECK: func.func private @_QPpass_assumed_size(!fir.ref<!fir.array<?xf32>>)
end module
