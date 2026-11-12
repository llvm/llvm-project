! Test -f[no-]backtrace driver forwarding to flang -fc1.
! An explicit -fbacktrace or -fno-backtrace is forwarded. The last flag wins.
! The default passes neither flag, and the frontend leaves the trace on.

! RUN: %flang -### %s 2>&1 | FileCheck %s --check-prefixes=FC1,DEFAULT
! RUN: %flang -### -fno-backtrace %s 2>&1 | FileCheck %s --check-prefixes=FC1,NO-BACKTRACE
! RUN: %flang -### -fbacktrace -fno-backtrace %s 2>&1 | FileCheck %s --check-prefixes=FC1,NO-BACKTRACE
! RUN: %flang -### -fbacktrace %s 2>&1 | FileCheck %s --check-prefixes=FC1,BACKTRACE
! RUN: %flang -### -fno-backtrace -fbacktrace %s 2>&1 | FileCheck %s --check-prefixes=FC1,BACKTRACE

! The frontend accepts both flags.
! RUN: %flang_fc1 -emit-hlfir -fbacktrace %s -o -
! RUN: %flang_fc1 -emit-hlfir -fno-backtrace %s -o -

! FC1: "-fc1"
! BACKTRACE-SAME: "-fbacktrace"
! NO-BACKTRACE-SAME: "-fno-backtrace"
! BACKTRACE-NOT: "-fno-backtrace"
! DEFAULT-NOT: "-fno-backtrace"
! DEFAULT-NOT: "-fbacktrace"

end program
