! Check that -ftime-report flag is passed as-is to fc1. The value of the flag
! is only checked there. This behavior intentionally mirrors that of clang.
!
! -ftime-report= is currently not supported because we do not support detailed
! timing information on the LLVM IR optimization and code generation passes.
! When that is supported, these can be re-enabled.
!
! XFAIL: *
!
! RUN: %flang -### -c -ftime-report=per-pass %s 2>&1 | FileCheck %s -check-prefix=PER-PASS
! RUN: %flang -### -c -ftime-report=per-pass-run %s 2>&1 | FileCheck %s -check-prefix=PER-PASS-INVOKE
! RUN: %flang -### -c -ftime-report=unknown %s 2>&1 | FileCheck %s -check-prefix=UNKNOWN

! PER-PASS: "-ftime-report=per-pass"
! PER-PASS-INVOKE: "-ftime-report=per-pass-run"
! UNKNOWN: "-ftime-report=unknown"

end program
