! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
module m
  procedure(func), pointer :: foo
  interface foo
     procedure :: foo
  end interface
 contains
  function func(x)
    func = x
  end
end

program main
  use m
!CHECK: foo => func
  foo => func
end
