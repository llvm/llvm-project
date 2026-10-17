! The driver-only checks (-### forwarding, no warning) live in
! flang/test/Driver/flang-f-opts.f90; this file checks the generated IR.

! Without the flag, an alwaysinline contained procedure is inlined and deleted.
! RUN: %flang_fc1 -emit-llvm -O2 -o - %s | FileCheck %s --check-prefix=DEFAULT \
! RUN:     --implicit-check-not="@_QFtest_alwaysPadd_two("
! -fkeep-inline-functions retains that definition, matching Clang's retention
! of inline functions owned by this translation unit.
! RUN: %flang_fc1 -emit-llvm -O2 -fkeep-inline-functions -o - %s | FileCheck %s --check-prefix=KEEP \
! RUN:     --implicit-check-not="@_QFtest_plainPadd_one("

subroutine test_always(n)
  integer :: n
  n = add_two(n)
contains
  integer function add_two(n)
    !dir$ inlinealways add_two
    integer :: n
    add_two = n + 2
  end function
end subroutine

subroutine test_plain(n)
  integer :: n
  n = add_one(n)
contains
  integer function add_one(n)
    integer :: n
    add_one = n + 1
  end function
end subroutine

! DEFAULT-LABEL: define void @test_always_(

! KEEP: @llvm.{{(compiler.)?}}used = {{.*}}@_QFtest_alwaysPadd_two
! KEEP-LABEL: define void @test_always_(
! KEEP-NOT: call {{.*}}@_QFtest_alwaysPadd_two(
! KEEP: define internal {{.*}}@_QFtest_alwaysPadd_two(
! A contained procedure that is not marked inline is still deleted.
! KEEP-LABEL: define void @test_plain_(
