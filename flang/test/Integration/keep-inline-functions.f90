! The driver-only checks (-### forwarding, no warning) live in
! flang/test/Driver/flang-f-opts.f90; this file checks the generated IR.

! Without the flag, an alwaysinline contained procedure is inlined and deleted.
! RUN: %flang_fc1 -emit-llvm -O2 -o - %s | FileCheck %s --check-prefix=DEFAULT \
! RUN:     --implicit-check-not="@_QFtest_alwaysPadd_two("
! -fkeep-inline-functions retains that definition, matching Clang's retention
! of inline functions owned by this translation unit.
! RUN: %flang_fc1 -emit-llvm -O2 -fkeep-inline-functions -o - %s | FileCheck %s --check-prefix=KEEP \
! RUN:     --implicit-check-not="@_QFtest_plainPadd_one("

! A procedure that is never called is still kept, and at every optimization
! level: the FIR inliner pass runs even at -O0 and would otherwise erase an
! uncalled procedure as dead code before it ever reaches LLVM IR, which is too
! late for the scan below to see it.
! RUN: %flang_fc1 -emit-llvm -O0 -o - %s | FileCheck %s --check-prefix=UNUSED-DEFAULT \
! RUN:     --implicit-check-not="@_QFtest_neverPunused("
! RUN: %flang_fc1 -emit-llvm -O0 -fkeep-inline-functions -o - %s | FileCheck %s --check-prefix=UNUSED-KEEP
! RUN: %flang_fc1 -emit-llvm -O2 -fkeep-inline-functions -o - %s | FileCheck %s --check-prefix=UNUSED-KEEP

! Check retention immediately after the FIR inliner, before LLVM IR
! translation. This needs the driver to forward MLIR diagnostic options.
! RUN: %flang -S -O0 -mmlir --mlir-disable-threading \
! RUN:     -mmlir --mlir-print-ir-after=inline %s -o /dev/null 2>&1 | \
! RUN:     FileCheck %s --check-prefix=MLIR-DEFAULT \
! RUN:     --implicit-check-not="@_QFtest_neverPunused"
! RUN: %flang -S -O0 -fkeep-inline-functions -mmlir --mlir-disable-threading \
! RUN:     -mmlir --mlir-print-ir-after=inline %s -o /dev/null 2>&1 | \
! RUN:     FileCheck %s --check-prefix=MLIR-KEEP

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

subroutine test_never(n)
  integer :: n
  n = n + 1
contains
  integer function unused(k)
    !dir$ inlinealways unused
    integer :: k
    unused = k - 1
  end function
end subroutine

! DEFAULT-LABEL: define void @test_always_(

! KEEP: @llvm.{{(compiler.)?}}used = {{.*}}@_QFtest_alwaysPadd_two
! KEEP-LABEL: define void @test_always_(
! KEEP-NOT: call {{.*}}@_QFtest_alwaysPadd_two(
! KEEP: define internal {{.*}}@_QFtest_alwaysPadd_two(
! A contained procedure that is not marked inline is still deleted.
! KEEP-LABEL: define void @test_plain_(

! UNUSED-DEFAULT-LABEL: define void @test_never_(

! UNUSED-KEEP: @llvm.{{(compiler.)?}}used = {{.*}}@_QFtest_neverPunused
! UNUSED-KEEP-LABEL: define void @test_never_(
! UNUSED-KEEP: define internal {{.*}}@_QFtest_neverPunused(

! MLIR-DEFAULT-LABEL: IR Dump After InlinerPass:
! MLIR-KEEP-LABEL: IR Dump After InlinerPass:
! MLIR-KEEP: func.func private @_QFtest_neverPunused{{.*}}attributes {{.*}}flang.keep_inline_functions_ref = @_QFtest_neverPunused{{.*}}llvm.always_inline
