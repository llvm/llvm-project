! RUN: bbc -fopenacc -fcuda -emit-hlfir %s -o - | FileCheck %s

! CUDA Fortran generic resolution treats objects listed in a structured
! !$acc data mapping clause as device addresses for DEVICE dummies, without
! changing ordinary host references in the same region.

module m
  interface doit
    subroutine __device_sub(a)
      real(4), device, intent(in) :: a(:,:,:)
      !dir$ ignore_tkr(c) a
    end
    subroutine __host_sub(a)
      real(4), intent(in) :: a(:,:,:)
      !dir$ ignore_tkr(c) a
    end
  end interface

  interface
    attributes(global) subroutine kernel(a, n)
      real(4), device :: a(*)
      integer, value :: n
    end
  end interface

  interface gemm
    subroutine gemm_dpm(alpha, a, b, beta, c)
      complex(8), device :: alpha, a(*), b(*), beta, c(*)
    end
    subroutine gemm_hpm(alpha, a, b, beta, c)
      complex(8) :: alpha, beta
      complex(8), device :: a(*), b(*), c(*)
    end
  end interface
end module

subroutine test_data_generic
  use m
  real(4) :: mapped(2,2,2)
  real(4) :: unmapped(2,2,2)
  call doit(mapped)
  !$acc data copyin(mapped)
  mapped(1,1,1) = 1.0
  call doit(mapped)
  call doit(unmapped)
  !$acc end data
  call doit(mapped)
end subroutine

! CHECK-LABEL: func.func @_QPtest_data_generic
! CHECK: %[[HOST:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_data_genericEmapped"}
! CHECK: fir.call @_QP__host_sub
! CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[HOST]]#0
! CHECK: acc.data dataOperands(%[[COPYIN]]
! CHECK: %[[DEV:.*]]:2 = hlfir.declare %[[COPYIN]]
! The host assignment keeps using the host address.
! CHECK: hlfir.designate %[[HOST]]#0
! CHECK: hlfir.assign
! The mapped actual selects the device specific and passes the device address.
! CHECK: %[[DEVBOX:.*]] = fir.embox %[[DEV]]#0
! CHECK: %[[DEVARG:.*]] = fir.convert %[[DEVBOX]]
! CHECK: fir.call @_QP__device_sub(%[[DEVARG]])
! An unmapped object stays host in the same region.
! CHECK: fir.call @_QP__host_sub
! CHECK: acc.terminator
! CHECK: fir.call @_QP__host_sub

subroutine test_data_hpm
  use m
  complex(8), parameter :: zero = (0.0d0, 0.0d0)
  complex(8), parameter :: one = (1.0d0, 0.0d0)
  complex(8) :: a(4), b(4), c(4)
  real(8) :: hvol
  hvol = 1.0d0
  !$acc data copyin(a, b) copyout(c)
  call gemm(one * hvol, a, b, zero, c)
  !$acc end data
end subroutine

! Host scalars stay host, so the host-pointer-mode specific wins over the
! one that also requires DEVICE alpha and beta.
! CHECK-LABEL: func.func @_QPtest_data_hpm
! CHECK: acc.data
! CHECK: fir.call @_QPgemm_hpm

! Mapping a named COMMON does not create a device binding for its members,
! so the host specific must still be selected.
subroutine test_common_block
  use m
  real(4) :: mapped(2,2,2)
  common /blk/ mapped
  !$acc data copyin(/blk/)
  call doit(mapped)
  !$acc end data
end subroutine

! CHECK-LABEL: func.func @_QPtest_common_block
! CHECK: acc.data
! CHECK: fir.call @_QP__host_sub
! CHECK-NOT: fir.call @_QP__device_sub

! A common member listed as a designator is a whole object, so it does get
! a device binding and the DEVICE specific.
subroutine test_common_member
  use m
  real(4) :: mapped(2,2,2)
  common /blk2/ mapped
  !$acc data copyin(mapped)
  call doit(mapped)
  !$acc end data
end subroutine

! CHECK-LABEL: func.func @_QPtest_common_member
! CHECK: acc.data
! CHECK: fir.call @_QP__device_sub

! A mapped component is not recorded as a whole-object device mapping, so the
! host specific is selected and the host address is passed.
subroutine test_component
  use m
  type t
    real(4) :: c(2,2,2)
  end type
  type(t) :: x
  !$acc data copyin(x%c)
  call doit(x%c)
  !$acc end data
end subroutine

! CHECK-LABEL: func.func @_QPtest_component
! CHECK: acc.data
! CHECK: fir.call @_QP__host_sub
! CHECK-NOT: fir.call @_QP__device_sub

! A kernel launch maps its own arguments, so it must keep the host address
! even though the DEVICE dummy would otherwise take the device binding.
subroutine test_kernel_launch
  use m
  real(4) :: mapped(100)
  !$acc data copyin(mapped)
  call kernel<<<1,1>>>(mapped, 100)
  !$acc end data
end subroutine

! CHECK-LABEL: func.func @_QPtest_kernel_launch
! CHECK: %[[HOST:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_kernel_launchEmapped"}
! CHECK: acc.data
! CHECK: %[[LAUNCH_ARG:.*]] = fir.convert %[[HOST]]#0
! CHECK: cuf.kernel_launch @_QPkernel{{.*}}(%[[LAUNCH_ARG]]
