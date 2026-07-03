!RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck --allow-empty %s
!CHECK-NOT: error:
program main
  sin = 1
  block
    intrinsic sin
    print *, sin(0.)
  end block
end
