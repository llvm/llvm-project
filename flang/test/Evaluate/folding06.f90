! RUN: %python %S/test_folding.py %s %flang_fc1
! Test transformational intrinsic function folding

module m

  ! Testing ASSOCATED
  integer, pointer :: int_pointer
  integer, allocatable :: int_allocatable
  logical, parameter :: test_Assoc1 = .not.(associated(null()))
  logical, parameter :: test_Assoc2 = .not.(associated(null(), null()))
  logical, parameter :: test_Assoc3 = .not.(associated(null(int_pointer)))
  logical, parameter :: test_Assoc4 = .not.(associated(null(int_allocatable)))
  logical, parameter :: test_Assoc5 = .not.(associated(null(), null(int_pointer)))
  logical, parameter :: test_Assoc6 = .not.(associated(null(), null(int_allocatable)))

  type A
    real(4) x
    integer(8) i
  end type

  integer(8), parameter :: new_shape(*) = [2, 4]
  integer(4), parameter :: order(2) = [2, 1]


  ! Testing integers (similar to real and complex)
  integer(4), parameter :: int_source(*) = [1, 2, 3, 4, 5, 6]
  integer(4), parameter :: int_pad(2) = [7, 8]
  integer(4), parameter :: int_expected_result(*, *) = reshape([1, 5, 2, 6, 3, 7, 4, 8], new_shape)
  integer(4), parameter :: int_result(*, *) = reshape(int_source, new_shape, int_pad, order)
  integer(4), parameter :: int_result_long_source(*, *) = reshape([1, 5, 2, 6, 3, 7, 4, 8, 9], new_shape)
  logical, parameter :: test_reshape_integer_1 = all(int_expected_result == int_result)
  logical, parameter :: test_reshape_integer_2 = all(shape(int_result, 8).EQ.new_shape)
  logical, parameter :: test_reshape_integer_3 = all(int_expected_result == int_result_long_source)


  ! Testing characters
  character(kind=1, len=3), parameter ::char_source(*) = ["abc", "def", "ghi", "jkl", "mno", "pqr"]
  character(kind=1,len=3), parameter :: char_pad(2) = ["stu", "vxy"]

  character(kind=1, len=3), parameter :: char_expected_result(*, *) = &
    reshape(["abc", "mno", "def", "pqr", "ghi", "stu", "jkl", "vxy"], new_shape)

  character(kind=1, len=3), parameter :: char_result(*, *) = &
    reshape(char_source, new_shape, char_pad, order)

  logical, parameter :: test_reshape_char_1 = all(char_result == char_expected_result)
  logical, parameter :: test_reshape_char_2 = all(shape(char_result, 8).EQ.new_shape)


  ! Testing derived types
  type(A), parameter :: derived_source(*) = &
    [A(x=1.5, i=1), A(x=2.5, i=2), A(x=3.5, i=3), A(x=4.5, i=4), A(x=5.5, i=5), A(x=6.5, i=6)]

  type(A), parameter :: derived_pad(2) = [A(x=7.5, i=7), A(x=8.5, i=8)]

  type(A), parameter :: derived_expected_result(*, *) = &
    reshape([a::a(x=1.5_4,i=1_8),a(x=5.5_4,i=5_8),a(x=2.5_4,i=2_8), a(x=6.5_4,i=6_8), &
      a(x=3.5_4,i=3_8),a(x=7.5_4,i=7_8),a(x=4.5_4,i=4_8),a(x=8.5_4,i=8_8)], new_shape)

  type(A), parameter :: derived_result(*, *) = reshape(derived_source, new_shape, derived_pad, order)

  logical, parameter :: test_reshape_derived_1 = all((derived_result%x.EQ.derived_expected_result%x) &
      .AND.(derived_result%i.EQ.derived_expected_result%i))

  logical, parameter :: test_reshape_derived_2 = all(shape(derived_result).EQ.new_shape)

  ! More complicated ORDER= arguments
  integer, parameter :: int3d(2,3,4) = reshape([(j,j=1,24)],shape(int3d))
  logical, parameter :: test_int3d = all([int3d] == [(j,j=1,24)])
  logical, parameter :: test_reshape_order_1 = all([reshape(int3d, [2,3,4], order=[1,2,3])] == [(j,j=1,24)])
  logical, parameter :: test_reshape_order_2 = all([reshape(int3d, [2,4,3], order=[1,3,2])] == [1,2,7,8,13,14,19,20,3,4,9,10,15,16,21,22,5,6,11,12,17,18,23,24])
  logical, parameter :: test_reshape_order_3 = all([reshape(int3d, [3,2,4], order=[2,1,3])] == [1,3,5,2,4,6,7,9,11,8,10,12,13,15,17,14,16,18,19,21,23,20,22,24])
  logical, parameter :: test_reshape_order_4 = all([reshape(int3d, [3,4,2], order=[2,3,1])] == [1,9,17,2,10,18,3,11,19,4,12,20,5,13,21,6,14,22,7,15,23,8,16,24])
  logical, parameter :: test_reshape_order_5 = all([reshape(int3d, [4,2,3], order=[3,1,2])] == [1,4,7,10,13,16,19,22,2,5,8,11,14,17,20,23,3,6,9,12,15,18,21,24])
  logical, parameter :: test_reshape_order_6 = all([reshape(int3d, [4,3,2], order=[3,2,1])] == [1,7,13,19,3,9,15,21,5,11,17,23,2,8,14,20,4,10,16,22,6,12,18,24])

end module
