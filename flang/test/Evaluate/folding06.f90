! RUN: %S/test_folding.sh %s %t %flang_fc1
! REQUIRES: shell
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
end module
