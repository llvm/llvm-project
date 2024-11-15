! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=50
! OpenMP Version 5.0
! 2.10.1 use_device_ptr clause
! List item in USE_DEVICE_PTR clause must not be structure element.
! List item in USE_DEVICE_PTR clause must be of type C_PTR.
! List items that appear in a use_device_ptr clause can not appear in
! use_device_addr clause.
! Same list item can not be present multiple times or in multipe
! USE_DEVICE_PTR clauses.

subroutine omp_target_data
   use iso_c_binding
   integer :: a(1024)
   type(C_PTR) :: b
   integer, pointer :: arrayB
   type my_type
    type(C_PTR) :: my_cptr
   end type my_type

   type(my_type) :: my_var
   a = 1

   !ERROR: A variable that is part of another variable (structure element) cannot appear on the TARGET DATA USE_DEVICE_PTR clause
   !$omp target data map(tofrom: a, arrayB) use_device_ptr(my_var%my_cptr)
      allocate(arrayB)
      call c_f_pointer(my_var%my_cptr, arrayB)
      a = arrayB
   !$omp end target data

   !WARNING: Use of non-C_PTR type 'a' in USE_DEVICE_PTR is deprecated, use USE_DEVICE_ADDR instead
   !$omp target data map(tofrom: a) use_device_ptr(a)
      a = 2
   !$omp end target data

   !ERROR: List item 'b' present at multiple USE_DEVICE_PTR clauses
   !$omp target data map(tofrom: a, arrayB) use_device_ptr(b) use_device_ptr(b)
      allocate(arrayB)
      call c_f_pointer(b, arrayB)
      a = arrayB
   !$omp end target data

   !ERROR: List item 'b' present at multiple USE_DEVICE_PTR clauses
   !$omp target data map(tofrom: a, arrayB) use_device_ptr(b,b)
      allocate(arrayB)
      call c_f_pointer(b, arrayB)
      a = arrayB
   !$omp end target data

   !ERROR: Variable 'b' may not appear on both USE_DEVICE_PTR and USE_DEVICE_ADDR clauses on a TARGET DATA construct
   !$omp target data map(tofrom: a, arrayB) use_device_addr(b) use_device_ptr(b)
      allocate(arrayB)
      call c_f_pointer(b, arrayB)
      a = arrayB
   !$omp end target data

   !ERROR: Variable 'b' may not appear on both USE_DEVICE_PTR and USE_DEVICE_ADDR clauses on a TARGET DATA construct
   !$omp target data map(tofrom: a, arrayB) use_device_ptr(b) use_device_addr(b)
      allocate(arrayB)
      call c_f_pointer(b, arrayB)
      a = arrayB
   !$omp end target data

end subroutine omp_target_data

