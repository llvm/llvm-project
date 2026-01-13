! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s

! CHECK: error: Unexpected syntax while parsing the statement-function statement
! CHECK: reala
program test_real
  reala(10)
end program

! CHECK: error: Unexpected syntax while parsing the statement-function statement
! CHECK: integerb
subroutine test_integer
  integerb(20)
end subroutine

! CHECK: error: Unexpected syntax while parsing the statement-function statement  
! CHECK: logicalflag
subroutine test_logical
  logicalflag
end subroutine

! CHECK: error: Unexpected syntax while parsing the statement-function statement
! CHECK: complexz
subroutine test_complex
  complexz
end subroutine

! CHECK: error: Unexpected syntax while parsing the statement-function statement
! CHECK: characterstr
subroutine test_character
  characterstr
end subroutine
