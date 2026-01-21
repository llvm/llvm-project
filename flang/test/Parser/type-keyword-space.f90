! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s

! CHECK: error: A blank space or a separator (::) is required after the type keyword
! CHECK: reala
program test_real
  reala(10)
end program

! CHECK: error: A blank space or a separator (::) is required after the type keyword
! CHECK: integerb
subroutine test_integer
  integerb(20)
end subroutine

! CHECK: error: A blank space or a separator (::) is required after the type keyword  
! CHECK: logicalflag
subroutine test_logical
  logicalflag
end subroutine

! CHECK: error: A blank space or a separator (::) is required after the type keyword
! CHECK: complexz
subroutine test_complex
  complexz
end subroutine

! CHECK: error: A blank space or a separator (::) is required after the type keyword
! CHECK: characterstr
subroutine test_character
  characterstr
end subroutine
