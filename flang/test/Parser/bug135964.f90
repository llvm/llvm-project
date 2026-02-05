! RUN: %flang_fc1 -fopenacc -fdebug-unparse %s | FileCheck %s
!CHECK: !$ACC ROUTINE(square) BIND(asdf)

function square(x)
  square = x * x
end

!$acc routine(square) bind(asdf)
