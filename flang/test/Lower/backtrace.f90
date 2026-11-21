! -fno-backtrace tells the runtime to disable the stack trace by setting
! FLANG_ERROR_BACKTRACE when that variable is not already set.
! The default and an explicit -fbacktrace do not emit a setting, and the
! runtime leaves tracing on.

! RUN: %flang_fc1 -emit-fir -o - %s | FileCheck --check-prefix=CHECK-DEFAULT %s
! RUN: %flang_fc1 -emit-fir -fbacktrace -o - %s | FileCheck --check-prefix=CHECK-DEFAULT %s
! RUN: %flang_fc1 -emit-fir -fno-backtrace -o - %s | FileCheck --check-prefix=CHECK-NO-BACKTRACE %s

program test
end

! CHECK-DEFAULT-NOT: FLANG_ERROR_BACKTRACE
! CHECK-NO-BACKTRACE: fir.string_lit "FLANG_ERROR_BACKTRACE\00"
! CHECK-NO-BACKTRACE: fir.string_lit "0\00"
