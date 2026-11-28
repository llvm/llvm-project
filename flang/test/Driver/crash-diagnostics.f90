! Test -fcrash-diagnostics option.

! This test requires ulimit
! UNSUPPORTED: system-windows, system-cygwin, system-solaris

! RUN: sh -c 'ulimit -s 100; %flang -fsyntax-only %s 2>&1 | FileCheck --check-prefix=REPRO %s'
! RUN: sh -c 'ulimit -s 100; %flang -fsyntax-only -fcrash-diagnostics=off %s 2>&1 | FileCheck --check-prefix=NO-REPRO %s'
! RUN: sh -c 'ulimit -s 100; %flang -fsyntax-only -fcrash-diagnostics=compiler %s 2>&1 | FileCheck --check-prefix=REPRO %s'
! RUN: sh -c 'ulimit -s 100; %flang -fsyntax-only -fcrash-diagnostics=all %s 2>&1 | FileCheck --check-prefix=REPRO %s'
! RUN: sh -c 'ulimit -s 100; %flang -fsyntax-only -fcrash-diagnostics=foo %s 2>&1 | FileCheck --check-prefix=REPRO %s'
! RUN: sh -c 'ulimit -s 100; %flang -fsyntax-only -fno-crash-diagnostics %s 2>&1 | FileCheck --check-prefix=NO-REPRO %s'

! REPRO: PLEASE ATTACH THE FOLLOWING CRASH REPRODUCER FILES TO THE BUG REPORT:
! NO-REPRO-NOT: PLEASE ATTACH THE FOLLOWING CRASH REPRODUCER FILES TO THE BUG REPORT:
