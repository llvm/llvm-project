!RUN: %flang_fc1 -fopenmp -fopenmp-version=60 -fdebug-dump-symbols %s | FileCheck %s

program omp_default_category
  use iso_c_binding
  type T
    integer :: x = 0
  end type

  integer, target :: scalar
  integer, pointer :: ptr
  type(c_ptr) :: cptr
  integer, target :: array(10)
  integer, pointer :: arrayptr(:)
  integer, allocatable :: darray(:)
  character :: char
  type(T) :: struct

  ptr => scalar
  cptr = c_loc(scalar)
  arrayptr => array
  allocate(darray(10))

  !$omp parallel default(private:scalar)
    !CHECK: array (OmpShared): HostAssoc
    !CHECK: scalar (OmpPrivate): HostAssoc
    array = 1
    scalar = 1
  !$omp end parallel

  !$omp parallel default(shared:scalar)
    !CHECK: array (OmpShared): HostAssoc
    !CHECK: scalar (OmpShared): HostAssoc
    array = 1
    scalar = 1
  !$omp end parallel
  !
  !$omp parallel default(firstprivate:scalar)
    !CHECK: array (OmpShared): HostAssoc
    !CHECK: scalar (OmpFirstPrivate): HostAssoc
    array = 1
    scalar = 1
  !$omp end parallel

  !$omp task default(private:scalar)
    !CHECK: array (OmpFirstPrivate, OmpImplicit): HostAssoc
    !CHECK: scalar (OmpPrivate): HostAssoc
    array = 1
    scalar = 1
  !$omp end task

  !$omp parallel default(private:aggregate)
    !CHECK: array (OmpPrivate): HostAssoc
    !CHECK: char (OmpPrivate): HostAssoc
    !CHECK: scalar (OmpShared): HostAssoc
    !CHECK: struct (OmpPrivate): HostAssoc
    array = 1
    char = 'A'
    scalar = 1
    struct%x = 1
  !$omp end parallel

  !$omp parallel default(private:pointer)
    !CHECK: arrayptr (OmpPrivate): HostAssoc
    !CHECK: ptr (OmpPrivate): HostAssoc
    !CHECK: scalar (OmpShared): HostAssoc
    ptr = 1
    arrayptr(1) = 1
    scalar = 1
  !$omp end parallel

  !$omp parallel default(private:allocatable)
    !CHECK: darray (OmpPrivate): HostAssoc
    !CHECK: scalar (OmpShared): HostAssoc
    scalar = 1
    darray = 1
  !$omp end parallel
end program omp_default_category
