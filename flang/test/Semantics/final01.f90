! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Test FINAL subroutine constraints C786-C789
module m1
  external :: external
  intrinsic :: sin
  real :: object
  procedure(valid), pointer :: pointer
  type :: parent(kind1, len1)
    integer, kind :: kind1 = 1
    integer, len :: len1 = 1
  end type
  type, extends(parent) :: child(kind2, len2)
    integer, kind :: kind2 = 2
    integer, len :: len2 = 2
   contains
    final :: valid
!ERROR: FINAL subroutine 'external' of derived type 'child' must be a module procedure
!ERROR: FINAL subroutine 'sin' of derived type 'child' must be a module procedure
!ERROR: FINAL subroutine 'object' of derived type 'child' must be a module procedure
!ERROR: FINAL subroutine 'pointer' of derived type 'child' must be a module procedure
!ERROR: FINAL subroutine 'func' of derived type 'child' must be a subroutine
    final :: external, sin, object, pointer, func
!ERROR: FINAL subroutine 's01' of derived type 'child' must have a single dummy argument that is a data object
!ERROR: FINAL subroutine 's02' of derived type 'child' must have a single dummy argument that is a data object
!ERROR: FINAL subroutine 's03' of derived type 'child' must not have a dummy argument with INTENT(OUT)
!ERROR: FINAL subroutine 's04' of derived type 'child' must not have a dummy argument with the VALUE attribute
!ERROR: FINAL subroutine 's05' of derived type 'child' must not have a POINTER dummy argument
!ERROR: FINAL subroutine 's06' of derived type 'child' must not have an ALLOCATABLE dummy argument
!ERROR: FINAL subroutine 's07' of derived type 'child' must not have a coarray dummy argument
!ERROR: FINAL subroutine 's08' of derived type 'child' must not have a polymorphic dummy argument
!ERROR: FINAL subroutine 's09' of derived type 'child' must not have a polymorphic dummy argument
!ERROR: FINAL subroutine 's10' of derived type 'child' must not have an OPTIONAL dummy argument
    final :: s01, s02, s03, s04, s05, s06, s07, s08, s09, s10
!ERROR: FINAL subroutine 's11' of derived type 'child' must have a single dummy argument
!ERROR: FINAL subroutine 's12' of derived type 'child' must have a single dummy argument
!ERROR: FINAL subroutine 's13' of derived type 'child' must have a dummy argument with an assumed LEN type parameter 'len1=*'
!ERROR: FINAL subroutine 's13' of derived type 'child' must have a dummy argument with an assumed LEN type parameter 'len2=*'
!ERROR: FINAL subroutine 's14' of derived type 'child' must have a dummy argument with an assumed LEN type parameter 'len2=*'
!ERROR: FINAL subroutine 's15' of derived type 'child' must have a dummy argument with an assumed LEN type parameter 'len1=*'
!ERROR: FINAL subroutine 's16' of derived type 'child' must not have a polymorphic dummy argument
!ERROR: FINAL subroutine 's17' of derived type 'child' must have a TYPE(child) dummy argument
    final :: s11, s12, s13, s14, s15, s16, s17
!ERROR: FINAL subroutine 'valid' already appeared in this derived type
    final :: valid
!ERROR: FINAL subroutines 'valid2' and 'valid' of derived type 'child' cannot be distinguished by rank or KIND type parameter value
    final :: valid2
  end type
 contains
  subroutine valid(x)
    type(child(len1=*, len2=*)), intent(inout) :: x
  end subroutine
  subroutine valid2(x)
    type(child(len1=*, len2=*)), intent(inout) :: x
  end subroutine
  real function func(x)
    type(child(len1=*, len2=*)), intent(inout) :: x
    func = 0.
  end function
  subroutine s01(*)
  end subroutine
  subroutine s02(x)
    external :: x
  end subroutine
  subroutine s03(x)
    type(child(kind1=3, len1=*, len2=*)), intent(out) :: x
  end subroutine
  subroutine s04(x)
    type(child(kind1=4, len1=*, len2=*)), value :: x
  end subroutine
  subroutine s05(x)
    type(child(kind1=5, len1=*, len2=*)), pointer :: x
  end subroutine
  subroutine s06(x)
    type(child(kind1=6, len1=*, len2=*)), allocatable :: x
  end subroutine
  subroutine s07(x)
    type(child(kind1=7, len1=*, len2=*)) :: x[*]
  end subroutine
  subroutine s08(x)
    class(child(kind1=8, len1=*, len2=*)) :: x
  end subroutine
  subroutine s09(x)
    class(*) :: x
  end subroutine
  subroutine s10(x)
    type(child(kind1=10, len1=*, len2=*)), optional :: x
  end subroutine
  subroutine s11(x, y)
    type(child(kind1=11, len1=*, len2=*)) :: x, y
  end subroutine
  subroutine s12
  end subroutine
  subroutine s13(x)
    type(child(kind1=13)) :: x
  end subroutine
  subroutine s14(x)
    type(child(kind1=14, len1=*,len2=2)) :: x
  end subroutine
  subroutine s15(x)
    type(child(kind1=15, len2=*)) :: x
  end subroutine
  subroutine s16(x)
    type(*) :: x
  end subroutine
  subroutine s17(x)
    type(parent(kind1=17, len1=*)) :: x
  end subroutine
  subroutine nested
    type :: t
     contains
!ERROR: FINAL subroutine 'internal' of derived type 't' must be a module procedure
      final :: internal
    end type
   contains
    subroutine internal(x)
      type(t), intent(inout) :: x
    end subroutine
  end subroutine
end module
