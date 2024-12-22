! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=50
! OpenMP Version 5.0
! 2.10.1 use_device_ptr clause
! List item in USE_DEVICE_ADDR clause must not be structure element.
! Same list item can not be present multiple times or in multipe
! USE_DEVICE_ADDR clauses.

subroutine omp_target_data
   integer :: a(1024)
   integer, target :: b(1024)
   type my_type
    integer :: my_b(1024)
   end type my_type

   type(my_type) :: my_var
   a = 1

   !ERROR: A variable that is part of another variable (structure element) cannot appear on the TARGET DATA USE_DEVICE_ADDR clause
   !$omp target data map(tofrom: a) use_device_addr(my_var%my_b)
      my_var%my_b = a
   !$omp end target data

   !ERROR: List item 'b' present at multiple USE_DEVICE_ADDR clauses
   !$omp target data map(tofrom: a) use_device_addr(b,b)
      b = a
   !$omp end target data

   !ERROR: List item 'b' present at multiple USE_DEVICE_ADDR clauses
   !$omp target data map(tofrom: a) use_device_addr(b) use_device_addr(b)
      b = a
   !$omp end target data

end subroutine omp_target_data
