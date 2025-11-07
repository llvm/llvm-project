!RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
!CHECK: TYPE(t) :: x = t(pp=f)
!CHECK-NOT: error:
interface
  function f()
  end
end interface
type t
  procedure(f), nopass, pointer :: pp
end type
type(t) :: x = t(pp=f)
end
