! RUN: %if x86-registered-target %{ %flang_fc1 -fopenmp \
! RUN:   -fopenmp-version=52 -triple x86_64-unknown-linux-gnu \
! RUN:   -emit-hlfir %s -o - | FileCheck %s %}

module source_context
contains
  subroutine depth_base
    !$omp declare variant(depth_vendor) &
    !$omp& match(implementation={vendor(score(3): llvm)})
    !$omp declare variant(depth_arch) match(device={arch(x86_64)})
  end subroutine
  subroutine depth_vendor
  end subroutine
  subroutine depth_arch
  end subroutine

  subroutine order_base
    !$omp declare variant(order_vendor) &
    !$omp& match(implementation={vendor(score(3): llvm)})
    !$omp declare variant(order_parallel) match(construct={parallel})
  end subroutine
  subroutine order_vendor
  end subroutine
  subroutine order_parallel
  end subroutine

  integer function value_base()
    !$omp declare variant(value_vendor) &
    !$omp& match(implementation={vendor(score(3): llvm)})
    !$omp declare variant(value_arch) match(device={arch(x86_64)})
    value_base = 0
  end function
  integer function value_vendor()
    value_vendor = 1
  end function
  integer function value_arch()
    value_arch = 2
  end function

! CHECK-LABEL: func.func @_QMsource_contextPtile_context(
! CHECK: fir.call @_QMsource_contextPdepth_arch()
! CHECK-NOT: fir.call @_QMsource_contextPdepth_vendor
! CHECK: return
  subroutine tile_context(n)
    integer :: n, i
    !$omp tile sizes(2)
    do i = 1, n
      call depth_base()
    end do
  end subroutine

! CHECK-LABEL: func.func @_QMsource_contextPunroll_context(
! CHECK: fir.call @_QMsource_contextPdepth_arch()
! CHECK-NOT: fir.call @_QMsource_contextPdepth_vendor
! CHECK: return
  subroutine unroll_context(n)
    integer :: n, i
    !$omp unroll partial(2)
    do i = 1, n
      call depth_base()
    end do
  end subroutine

! CHECK-LABEL: func.func @_QMsource_contextPfuse_context(
! CHECK-NOT: fir.call @_QMsource_contextPdepth_vendor
! CHECK-COUNT-2: fir.call @_QMsource_contextPdepth_arch()
! CHECK-NOT: fir.call @_QMsource_contextPdepth_vendor
! CHECK: return
  subroutine fuse_context(n)
    integer :: n, i, j
    !$omp fuse
    do i = 1, n
      call depth_base()
    end do
    do j = 1, n
      call depth_base()
    end do
    !$omp end fuse
  end subroutine

! CHECK-LABEL: func.func @_QMsource_contextPatomic_context(
! CHECK: fir.call @_QMsource_contextPvalue_arch()
! CHECK-NOT: fir.call @_QMsource_contextPvalue_vendor
! CHECK: omp.atomic.update
! CHECK: return
  subroutine atomic_context(x)
    integer :: x
    !$omp atomic update
    x = x + value_base()
  end subroutine

! CHECK-LABEL: func.func @_QMsource_contextPassume_context()
! CHECK: fir.call @_QMsource_contextPdepth_vendor()
! CHECK-NOT: fir.call @_QMsource_contextPdepth_arch
! CHECK: return
  subroutine assume_context
    !$omp assume holds(.true.)
      call depth_base()
    !$omp end assume
  end subroutine

! CHECK-LABEL: func.func @_QMsource_contextPcombined_context(
! CHECK: fir.call @_QMsource_contextPorder_parallel()
! CHECK-NOT: fir.call @_QMsource_contextPorder_vendor
! CHECK: return
  subroutine combined_context(n)
    integer :: n, i
    !$omp teams distribute parallel do
    do i = 1, n
      call order_base()
    end do
  end subroutine
end module
