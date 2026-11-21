! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

module target_polymorphic_derived_type_version
  implicit none

  type :: base_t
    integer :: x = 0
  contains
    procedure :: set_x
  end type

  type, extends(base_t) :: child_t
    integer :: y = 0
  end type

contains
  subroutine set_x(this)
    class(base_t), intent(inout) :: this
    this%x = 1
  end subroutine

  subroutine map_class_object()
    class(base_t), allocatable :: obj
    type(base_t) :: nonpoly

    allocate(child_t :: obj)

    ! TYPE(dt) is still valid before OpenMP 6.1.
    !$omp target map(tofrom: nonpoly)
      nonpoly%x = 1
    !$omp end target

    !ERROR: Polymorphic type 'obj' may not appear in a MAP clause on a target offload construct before OpenMP 6.1
    !$omp target map(tofrom: obj)
    !$omp end target

    !ERROR: Polymorphic type 'obj' may not appear in a MAP clause on a target offload construct before OpenMP 6.1
    !$omp target enter data map(to: obj)

    !ERROR: Polymorphic type 'obj' may not appear in a MAP clause on a target offload construct before OpenMP 6.1
    !$omp target exit data map(delete: obj)
  end subroutine

  subroutine map_unlimited_polymorphic_object()
    class(*), allocatable :: any
    type(base_t) :: nonpoly

    allocate(child_t :: any)

    ! TYPE(dt) is still valid before OpenMP 6.1.
    !$omp target map(tofrom: nonpoly)
      nonpoly%x = 1
    !$omp end target

    !ERROR: Polymorphic type 'any' may not appear in a MAP clause on a target offload construct before OpenMP 6.1
    !$omp target map(tofrom: any)
    !$omp end target

    !ERROR: Polymorphic type 'any' may not appear in a MAP clause on a target offload construct before OpenMP 6.1
    !$omp target data map(tofrom: any)
    !$omp end target data

    !ERROR: Polymorphic type 'any' may not appear in a MAP clause on a target offload construct before OpenMP 6.1
    !$omp target enter data map(to: any)

    !ERROR: Polymorphic type 'any' may not appear in a MAP clause on a target offload construct before OpenMP 6.1
    !$omp target exit data map(delete: any)
  end subroutine

  subroutine map_class_component()
    class(base_t), allocatable :: obj

    allocate(child_t :: obj)

    !ERROR: Polymorphic type 'obj' may not appear in a MAP clause on a target offload construct before OpenMP 6.1
    !$omp target map(tofrom: obj%x)
    !$omp end target
  end subroutine

  subroutine target_region_use()
    class(base_t), allocatable :: obj
    type(base_t) :: nonpoly

    allocate(child_t :: obj)

    !$omp target
      nonpoly%x = 2
    !$omp end target

    !$omp target
      !ERROR: Polymorphic type 'obj' may not be used in a target region before OpenMP 6.1
      obj%x = 3
    !$omp end target
  end subroutine

  subroutine target_region_unlimited_polymorphic_use()
    class(*), allocatable :: any
    type(base_t) :: nonpoly

    allocate(child_t :: any)

    !$omp target
      nonpoly%x = 2
    !$omp end target

    !$omp target
      !ERROR: Polymorphic type 'any' may not be used in a target region before OpenMP 6.1
      if (same_type_as(any, nonpoly)) nonpoly%x = 3
    !$omp end target
  end subroutine
end module
