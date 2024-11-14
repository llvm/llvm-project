! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
module m
 contains
  !CHECK: expected end of statement
  !CHECK: subroutine s1(var i, j)
  subroutine s1(var i, j)
  end subroutine
  !CHECK: expected end of statement
  !CHECK: subroutine s2[b]
  subroutine s2[b]
  end subroutine
  !CHECK: expected end of statement
  !CHECK: function f1(var i, j)
  function f1(var i, j)
  end function
  !CHECK: expected end of statement
  !CHECK: function f2[b]
  function f2[b]
  end function
  !CHECK: expected end of statement
  !CHECK: function f3(a,*)
  function f3(a,*)
  end function
end
