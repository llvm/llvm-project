! This test checks correct propagation of location information in OpenACC
! operations.


! RUN: bbc -fopenacc -emit-hlfir --mlir-print-debuginfo --mlir-print-local-scope %s -o - | FileCheck %s
module acc_locations
  implicit none

  contains

  subroutine standalone_data_directive_locations(arr)
    real, dimension(10) :: arr

    !$acc enter data create(arr)
    !CHECK-LABEL: acc.enter_data
    !CHECK-SAME:  loc("{{.*}}locations.f90":14:11)

    !$acc update device(arr)
    !CHECK-LABEL: acc.update_device varPtr
    !CHECK-SAME:  loc("{{.*}}locations.f90":18:25)
    !CHECK-LABEL: acc.update dataOperands
    !CHECK-SAME:  loc("{{.*}}locations.f90":18:11)

    !$acc update host(arr)
    !CHECK-LABEL: acc.getdeviceptr varPtr
    !CHECK-SAME:  loc("{{.*}}locations.f90":24:23)
    !CHECK-LABEL: acc.update dataOperands
    !CHECK-SAME:  loc("{{.*}}locations.f90":24:11)
    !CHECK-LABEL: acc.update_host
    !CHECK-SAME:  loc("{{.*}}locations.f90":24:23)

    !$acc exit data delete(arr)
    !CHECK-LABEL: acc.exit_data
    !CHECK-SAME:  loc("{{.*}}locations.f90":32:11)

  end subroutine

  subroutine nested_acc_locations(arr1d)
    real, dimension(10) :: arr1d
    integer :: i

    !$acc data copy(arr1d)
    !$acc parallel
    !$acc loop
    do i = 1, 10
      arr1d(i) = arr1d(i) * 2
    end do
    !$acc end parallel
    !$acc end data

    !CHECK: acc.data
    !CHECK: acc.parallel
    !CHECK: acc.loop

    !CHECK:        acc.yield loc("{{.*}}locations.f90":44:11)
    !CHECK-NEXT: } attributes {{.*}} loc(fused["{{.*}}locations.f90":44:11, "{{.*}}locations.f90":45:5])

    !CHECK:        acc.yield loc("{{.*}}locations.f90":43:11)
    !CHECK-NEXT: } loc("{{.*}}locations.f90":43:11)

    !CHECK-NEXT:   acc.terminator loc("{{.*}}locations.f90":42:11)
    !CHECK-NEXT: } loc("{{.*}}locations.f90":42:11)

  end subroutine

  subroutine runtime_directive()

    !$acc init
    !CHECK-LABEL: acc.init
    !CHECK-SAME:  loc("{{.*}}locations.f90":68:11)

    !$acc shutdown
    !CHECK-LABEL: acc.shutdown
    !CHECK-SAME:  loc("{{.*}}locations.f90":72:11)

  end subroutine

  subroutine combined_directive_locations(arr)
    real :: arr(:)
    integer :: i

    !$acc parallel loop
    do i = 1, size(arr)
      arr(i) = arr(i) * arr(i)
    end do

    !CHECK: acc.parallel
    !CHECK: acc.loop
    !CHECK:      acc.yield loc("{{.*}}locations.f90":82:11)
    !CHECK-NEXT: } {{.*}} loc(fused["{{.*}}locations.f90":82:11, "{{.*}}locations.f90":83:5])
    !CHECK:      acc.yield loc("{{.*}}locations.f90":82:11)
    !CHECK-NEXT: } loc("{{.*}}locations.f90":82:11)
  end subroutine

  subroutine if_clause_expr_location(arr)
    real :: arr(:)
    integer :: i

    !$acc parallel loop if(.true.)
    do i = 1, size(arr)
      arr(i) = arr(i) * arr(i)
    end do

    !CHECK: %{{.*}} = arith.constant true loc("{{.*}}locations.f90":99:25)

    !CHECK: acc.parallel
    !CHECK: acc.loop
    !CHECK:      acc.yield loc("{{.*}}locations.f90":99:11)
    !CHECK-NEXT: } {{.*}} loc(fused["{{.*}}locations.f90":99:11, "{{.*}}locations.f90":100:5])
    !CHECK:      acc.yield loc("{{.*}}locations.f90":99:11)
    !CHECK-NEXT: } loc("{{.*}}locations.f90":99:11)
  end subroutine

  subroutine atomic_read_loc()
    integer(4) :: x
    integer(8) :: y
  
    !$acc atomic read
    y = x
  end
  !CHECK: acc.atomic.read {{.*}} loc("{{.*}}locations.f90":118:11)

  subroutine atomic_capture_loc()
    implicit none
    integer :: k, v, i
  
    k = 1
    v = 0
  
    !$acc atomic capture
    v = k
    k = (i + 1) * 3.14
    !$acc end atomic

! CHECK: acc.atomic.capture {
! CHECK:   acc.atomic.read {{.*}} loc("{{.*}}locations.f90":130:11)
! CHECK:   acc.atomic.write {{.*}} loc("{{.*}}locations.f90":130:11)
! CHECK: } loc("{{.*}}locations.f90":130:11)

  end subroutine

  subroutine atomic_update_loc()
    implicit none
    integer :: x, y, z
    
    !$acc atomic 
    y = y + 1
! CHECK: acc.atomic.update %{{.*}} : !fir.ref<i32> {
! CHECK: ^bb0(%{{.*}}: i32 loc("{{.*}}locations.f90":142:3)):
! CHECK: } loc("{{.*}}locations.f90":142:3)
    
    !$acc atomic update
    z = x * z
  end subroutine

  subroutine acc_loop_fused_locations(arr)
    real, dimension(10,10,10) :: arr
    integer :: i, j, k

    !$acc loop collapse(3)
    do i = 1, 10
      do j = 1, 10
        do k = 1, 10
          arr(i,j,k) = arr(i,j,k) * 2
        end do
      end do
    end do
  end subroutine

! CHECK-LABEL: func.func @_QMacc_locationsPacc_loop_fused_locations
! CHECK: acc.loop
! CHECK: } attributes {collapse = [3]{{.*}}} loc(fused["{{.*}}locations.f90":160:11, "{{.*}}locations.f90":161:5, "{{.*}}locations.f90":162:7, "{{.*}}locations.f90":163:9])

  subroutine data_end_locations(arr)
    real, dimension(10) :: arr

    !$acc data copy(arr)
    !CHECK-LABEL: acc.copyin
    !CHECK-SAME:  loc("{{.*}}locations.f90":177:21)

    !$acc end data
    !CHECK-LABEL: acc.copyout
    !CHECK-SAME:  loc("{{.*}}locations.f90":181:11)
  end subroutine
end module


