! This test verifies that both invocations produce a consistent order in the
! generated `.mod` file. Previous versions of Flang exhibited non-deterministic
! behavior due to pointers outside the cooked source being used to order symbols
! in the `.mod` file.

! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang_fc1 -fsyntax-only -J%t %S/Inputs/modfile72.f90
! RUN: %flang_fc1 -fsyntax-only -J%t %s
! RUN: cat %t/bar.mod | FileCheck %s

! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang_fc1 -fsyntax-only -J%t %S/Inputs/modfile72.f90 %s
! RUN: cat %t/bar.mod | FileCheck %s

module bar
  use foo, only : do_foo
  use foo, only : do_bar
contains
  subroutine do_baz()
    call do_foo()
    call do_bar()
  end
end

!      CHECK: use foo,only:do_foo
! CHECK-NEXT: use foo,only:do_bar
! CHECK-NEXT: use foo,only:foo$foo$do_bar_impl=>do_bar_impl
! CHECK-NEXT: use foo,only:foo$foo$do_foo_impl=>do_foo_impl
