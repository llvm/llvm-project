! RUN: %flang -### -fkeep-inline-functions %s 2>&1 | FileCheck %s --check-prefix=DRIVER-KEEP
! RUN: %flang -### -fno-keep-inline-functions %s 2>&1 | FileCheck %s --check-prefix=DRIVER-NOKEEP
! RUN: %flang -### -fkeep-inline-functions -fno-keep-inline-functions %s 2>&1 | FileCheck %s --check-prefix=DRIVER-NOKEEP
! RUN: %flang -### -fno-keep-inline-functions -fkeep-inline-functions %s 2>&1 | FileCheck %s --check-prefix=DRIVER-KEEP

! DRIVER-KEEP: "-fc1"
! DRIVER-KEEP-SAME: "-fkeep-inline-functions"
! DRIVER-NOKEEP: "-fc1"
! DRIVER-NOKEEP-NOT: "-fkeep-inline-functions"
! DRIVER-NOKEEP-NOT: "-fno-keep-inline-functions"

! Without the flag, an alwaysinline contained procedure is inlined and deleted.
! RUN: %flang_fc1 -emit-llvm -O2 -o - %s | FileCheck %s --check-prefix=DEFAULT
! -fkeep-inline-functions retains that definition, matching Clang's retention
! of inline functions owned by this translation unit.
! RUN: %flang_fc1 -emit-llvm -O2 -fkeep-inline-functions -o - %s | FileCheck %s --check-prefix=KEEP

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
! DEFAULT-NOT: @_QFtest_alwaysPadd_two(
! DEFAULT-NOT: define {{.*}}@_QFtest_alwaysPadd_two(

! KEEP: @llvm.{{(compiler.)?}}used = {{.*}}@_QFtest_alwaysPadd_two
! KEEP-LABEL: define void @test_always_(
! KEEP-NOT: call {{.*}}@_QFtest_alwaysPadd_two(
! KEEP: define internal {{.*}}@_QFtest_alwaysPadd_two(
! A contained procedure that is not marked inline is still deleted.
! KEEP-LABEL: define void @test_plain_(
! KEEP-NOT: @_QFtest_plainPadd_one(
! KEEP-NOT: define {{.*}}@_QFtest_plainPadd_one(
