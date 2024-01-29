! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.1
! Check OpenMP construct validity for the following directives:
! 2.14.7 Declare Target Directive

module declare_target01
  use omp_lib
  type my_type(kind_param, len_param)
    integer, KIND :: kind_param
    integer, LEN :: len_param
    integer :: t_i
    integer :: t_arr(10)
  end type my_type

  type(my_type(2, 4)) :: my_var, my_var2
  integer :: arr(10), arr2(10)
  integer(kind=4) :: x, x2
  character(len=32) :: w, w2
  integer, dimension(:), allocatable :: y, y2

  !$omp declare target (my_var)

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the DECLARE TARGET directive
  !$omp declare target (my_var%t_i)

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the DECLARE TARGET directive
  !$omp declare target (my_var%t_arr)

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !$omp declare target (my_var%kind_param)

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !$omp declare target (my_var%len_param)

  !$omp declare target (arr)

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the DECLARE TARGET directive
  !$omp declare target (arr(1))

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the DECLARE TARGET directive
  !$omp declare target (arr(1:2))

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !$omp declare target (x%KIND)

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !$omp declare target (w%LEN)

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !$omp declare target (y%KIND)

  !WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead.
  !$omp declare target to (my_var)

  !$omp declare target enter (my_var)

  !WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead.
  !$omp declare target to (my_var) device_type(host)
  
  !$omp declare target enter (my_var) device_type(host)

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the DECLARE TARGET directive
  !WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead.
  !$omp declare target to (my_var%t_i)

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the DECLARE TARGET directive
  !$omp declare target enter (my_var%t_i)

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the DECLARE TARGET directive
  !WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead.
  !$omp declare target to (my_var%t_arr)

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the DECLARE TARGET directive
  !$omp declare target enter (my_var%t_arr)

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead.
  !$omp declare target to (my_var%kind_param)

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !$omp declare target enter (my_var%kind_param)

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead.
  !$omp declare target to (my_var%len_param)

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !$omp declare target enter (my_var%len_param)

  !WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead.
  !$omp declare target to (arr)

  !$omp declare target enter (arr)

  !WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead.
  !$omp declare target to (arr) device_type(nohost)

  !$omp declare target enter (arr) device_type(nohost)

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the DECLARE TARGET directive
  !WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead.
  !$omp declare target to (arr(1))

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the DECLARE TARGET directive
  !$omp declare target enter (arr(1))

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the DECLARE TARGET directive
  !WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead.
  !$omp declare target to (arr(1:2))

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the DECLARE TARGET directive
  !$omp declare target enter (arr(1:2))

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead.
  !$omp declare target to (x%KIND)

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !$omp declare target enter (x%KIND)

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead.
  !$omp declare target to (w%LEN)

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !$omp declare target enter (w%LEN)

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead.
  !$omp declare target to (y%KIND)

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !$omp declare target enter (y%KIND)

  !$omp declare target link (my_var2)

  !$omp declare target link (my_var2) device_type(any)

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the DECLARE TARGET directive
  !$omp declare target link (my_var2%t_i)

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the DECLARE TARGET directive
  !$omp declare target link (my_var2%t_arr)

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !$omp declare target link (my_var2%kind_param)

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !$omp declare target link (my_var2%len_param)

  !$omp declare target link (arr2)

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the DECLARE TARGET directive
  !$omp declare target link (arr2(1))

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the DECLARE TARGET directive
  !$omp declare target link (arr2(1:2))

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !$omp declare target link (x2%KIND)

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !$omp declare target link (w2%LEN)

  !ERROR: A type parameter inquiry cannot appear on the DECLARE TARGET directive
  !$omp declare target link (y2%KIND)
end
