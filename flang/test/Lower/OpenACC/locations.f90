! This test checks correct propagation of location information in OpenACC
! operations.

! RUN: bbc -fopenacc -emit-fir --mlir-print-debuginfo --mlir-print-local-scope %s -o - | FileCheck %s

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
    !CHECK-NEXT: } loc("{{.*}}locations.f90":44:11)

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
    !CHECK-NEXT: } loc("{{.*}}locations.f90":82:11)
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
    !CHECK-NEXT: } loc("{{.*}}locations.f90":99:11)
    !CHECK:      acc.yield loc("{{.*}}locations.f90":99:11)
    !CHECK-NEXT: } loc("{{.*}}locations.f90":99:11)
  end subroutine

end module
