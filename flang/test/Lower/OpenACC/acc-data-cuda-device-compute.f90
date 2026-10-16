! RUN: bbc -fopenacc -fcuda -emit-hlfir %s -o - | FileCheck %s

! Calls inside compute constructs must use the ordinary symbol binding, like
! other references in the construct. A DEVICE dummy must not override it with
! the alternate binding of an enclosing data construct.
module compute_calls
  interface
    attributes(device) subroutine device_sub(a)
      real :: a(*)
    end subroutine
    subroutine host_sub(a)
      real, device :: a(*)
    end subroutine
    attributes(device) subroutine device_c(a) bind(c)
      real :: a(*)
    end subroutine
  end interface
contains
  subroutine implicit_mapping(a, n)
    real :: a(100)
    integer :: n, i
    !$acc data copy(a)
    call host_sub(a)
    !$acc parallel
    if (n > 0) call device_sub(a)
    !$acc end parallel
    !$acc serial
    call device_sub(a)
    !$acc end serial
    !$acc kernels
    call device_sub(a)
    !$acc end kernels
    !$acc parallel loop
    do i = 1, n
      call device_sub(a)
    end do
    !$acc end parallel loop
    !$acc serial loop
    do i = 1, n
      call device_sub(a)
    end do
    !$acc end serial loop
    !$acc kernels loop
    do i = 1, n
      call device_sub(a)
    end do
    !$acc end kernels loop
    call host_sub(a)
    !$acc end data
  end subroutine

! CHECK-LABEL: func.func @_QMcompute_callsPimplicit_mapping
! CHECK: %[[HOST:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QMcompute_callsFimplicit_mappingEa")
! CHECK: %[[COPY:.*]] = acc.copyin varPtr(%[[HOST]]#0
! CHECK: acc.data dataOperands(%[[COPY]]
! CHECK: %[[DEVICE:.*]]:2 = hlfir.declare %[[COPY]]
! CHECK: %[[ARG:.*]] = fir.convert %[[DEVICE]]#0
! CHECK: fir.call @_QPhost_sub(%[[ARG]])
! CHECK: acc.parallel
! CHECK: fir.if
! CHECK: %[[ARG:.*]] = fir.convert %[[HOST]]#0
! CHECK: fir.call @_QPdevice_sub(%[[ARG]])
! CHECK: acc.serial
! CHECK: %[[ARG:.*]] = fir.convert %[[HOST]]#0
! CHECK: fir.call @_QPdevice_sub(%[[ARG]])
! CHECK: acc.kernels
! CHECK: %[[ARG:.*]] = fir.convert %[[HOST]]#0
! CHECK: fir.call @_QPdevice_sub(%[[ARG]])
! CHECK: acc.parallel combined(loop)
! CHECK: acc.loop
! CHECK: %[[ARG:.*]] = fir.convert %[[HOST]]#0
! CHECK: fir.call @_QPdevice_sub(%[[ARG]])
! CHECK: acc.serial combined(loop)
! CHECK: acc.loop
! CHECK: %[[ARG:.*]] = fir.convert %[[HOST]]#0
! CHECK: fir.call @_QPdevice_sub(%[[ARG]])
! CHECK: acc.kernels combined(loop)
! CHECK: acc.loop
! CHECK: %[[ARG:.*]] = fir.convert %[[HOST]]#0
! CHECK: fir.call @_QPdevice_sub(%[[ARG]])
! CHECK: %[[ARG:.*]] = fir.convert %[[DEVICE]]#0
! CHECK: fir.call @_QPhost_sub(%[[ARG]])

  subroutine explicit_mapping(a)
    real :: a(100)
    !$acc data copy(a)
    !$acc parallel present(a)
    call device_sub(a)
    !$acc end parallel
    !$acc serial private(a)
    call device_sub(a)
    !$acc end serial
    !$acc kernels present(a)
    call device_sub(a)
    !$acc end kernels
    !$acc end data
  end subroutine

! CHECK-LABEL: func.func @_QMcompute_callsPexplicit_mapping
! CHECK: %[[HOST:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QMcompute_callsFexplicit_mappingEa")
! CHECK: acc.data
! CHECK: %[[PRESENT:.*]] = acc.present varPtr(%[[HOST]]#0
! CHECK: acc.parallel dataOperands(%[[PRESENT]]
! CHECK: %[[LOCAL:.*]]:2 = hlfir.declare %[[PRESENT]]
! CHECK: %[[ARG:.*]] = fir.convert %[[LOCAL]]#0
! CHECK: fir.call @_QPdevice_sub(%[[ARG]])
! CHECK: %[[PRIVATE:.*]] = acc.private varPtr(%[[HOST]]#0
! CHECK: acc.serial private(
! CHECK: %[[LOCAL:.*]]:2 = hlfir.declare %[[PRIVATE]]
! CHECK: %[[ARG:.*]] = fir.convert %[[LOCAL]]#0
! CHECK: fir.call @_QPdevice_sub(%[[ARG]])
! CHECK: %[[PRESENT:.*]] = acc.present varPtr(%[[HOST]]#0
! CHECK: acc.kernels dataOperands(%[[PRESENT]]
! CHECK: %[[LOCAL:.*]]:2 = hlfir.declare %[[PRESENT]]
! CHECK: %[[ARG:.*]] = fir.convert %[[LOCAL]]#0
! CHECK: fir.call @_QPdevice_sub(%[[ARG]])

  subroutine allocatable_mapping(a)
    real, allocatable :: a(:)
    !$acc data copy(a)
    !$acc parallel
    call device_c(a)
    !$acc end parallel
    !$acc end data
  end subroutine

! CHECK-LABEL: func.func @_QMcompute_callsPallocatable_mapping
! CHECK: %[[HOST:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QMcompute_callsFallocatable_mappingEa")
! CHECK: acc.data
! CHECK: acc.parallel
! CHECK: %[[BOX:.*]] = fir.load %[[HOST]]#0
! CHECK: %[[ADDR:.*]] = fir.box_addr %[[BOX]]
! CHECK: %[[ARG:.*]] = fir.convert %[[ADDR]]
! CHECK: fir.call @device_c(%[[ARG]])
end module
