! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! Test parsing of alternative order of char-length in an
! entity-decl or component-decl.
program p
  type t
    !CHECK: CHARACTER c1(2_4)*3/"abc", "def"/
    character c1*3(2)/'abc','def'/
  end type
  integer, parameter :: n=3
  !CHECK: CHARACTER v1(2_4)*(3_4)/"ghi", "jkl"/
  character v1*(n)(2)/'ghi','jkl'/
  !CHECK: CHARACTER :: v2(1_4)*2 = "mn"
  character::v2*2(1)='mn'
end



