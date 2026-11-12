! RUN: %python %S/test_symbols.py %s %flang_fc1
! Ensure that global ENTRY symbols with global bindings
! are hidden in distinct global scopes, and nothing
! clashes so long as binding names are distinct.

!DEF: /s1 (Subroutine) Subprogram
subroutine s1
 !DEF: /foo (Subroutine) Subprogram
 entry foo()
end subroutine
!DEF: /s2 (Subroutine) Subprogram
subroutine s2
 !DEF: /foo BIND(C) (Subroutine) Subprogram
 entry foo() bind(c, name="foo1")
end subroutine
!DEF: /s3 (Subroutine) Subprogram
subroutine s3
 !DEF: /foo BIND(C) (Subroutine) Subprogram
 entry foo() bind(c, name="foo2")
end subroutine
