!RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
external s
type t
  procedure(), nopass, pointer :: p => s
end type
!CHECK: TYPE(t) :: x = t(p=s)
type(t) :: x = t()
end
