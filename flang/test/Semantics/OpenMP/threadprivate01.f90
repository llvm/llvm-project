! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.1
! Check OpenMP construct validity for the following directives:
! 2.21.2 Threadprivate Directive

module thread_private01
  use omp_lib
  type my_type(kind_param, len_param)
    integer, KIND :: kind_param
    integer, LEN :: len_param
    integer :: t_i
    integer :: t_arr(10)
  end type my_type

  type(my_type(2, 4)) :: my_var
  integer :: arr(10)
  integer(kind=4) :: x
  character(len=32) :: w
  integer, dimension(:), allocatable :: y

  !$omp threadprivate(my_var)

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the THREADPRIVATE directive
  !$omp threadprivate(my_var%t_i)

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the THREADPRIVATE directive
  !$omp threadprivate(my_var%t_arr)

  !ERROR: A type parameter inquiry cannot appear on the THREADPRIVATE directive
  !$omp threadprivate(my_var%kind_param)

  !ERROR: A type parameter inquiry cannot appear on the THREADPRIVATE directive
  !$omp threadprivate(my_var%len_param)

  !$omp threadprivate(arr)

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the THREADPRIVATE directive
  !$omp threadprivate(arr(1))

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the THREADPRIVATE directive
  !$omp threadprivate(arr(1:2))

  !ERROR: A type parameter inquiry cannot appear on the THREADPRIVATE directive
  !$omp threadprivate(x%KIND)

  !ERROR: A type parameter inquiry cannot appear on the THREADPRIVATE directive
  !$omp threadprivate(w%LEN)

  !ERROR: A type parameter inquiry cannot appear on the THREADPRIVATE directive
  !$omp threadprivate(y%KIND)
end
