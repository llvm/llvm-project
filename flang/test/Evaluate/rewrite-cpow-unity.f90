! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
subroutine s(z, w)
  complex :: z
  complex(8) :: w
!CHECK: CALL foo((z))
  call foo(z**(1., 0.))
!CHECK: CALL foo((w))
  call foo(w**(1._8, -0._8))
!CHECK: CALL foo(z**(1._4,1._4))
  call foo(z**(1., 1.))
end
