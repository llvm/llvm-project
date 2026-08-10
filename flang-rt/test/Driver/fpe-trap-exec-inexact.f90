! Test that -ffpe-trap=inexact enables run-time halting on an IEEE_INEXACT
! exception.

! Halting requires both glibc (feenableexcept) and hardware that delivers a
! trap when an enabled FP exception is raised. Arm makes FP-exception trapping
! optional, and the AArch64 CI hardware does not deliver SIGFPE, so restrict
! this run-time test to x86 glibc.
! REQUIRES: target=x86_64{{.*}}-linux-gnu
! UNSUPPORTED: offload-cuda

! Built without traps: the inexact division completes and the program exits
! normally.
! RUN: %flang %isysroot -L"%libdir" %s -o %t.notrap
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t.notrap \
! RUN:     | FileCheck --check-prefix=NOTRAP %s

! Built with -ffpe-trap=inexact: the division 1.0/3.0 is inexact and halting
! terminates the program with SIGFPE (signal 8, i.e. shell exit 136). This is
! the first floating-point operation, so the trap is deterministic.
! RUN: %flang %isysroot -L"%libdir" -ffpe-trap=inexact %s -o %t.trap
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" sh -c 'ulimit -c 0; %t.trap; test $? -eq 136'

program fpe_trap_inexact
  real :: x, y
  ! real(3 + count) is exact for the runtime-unknown small integer, so the
  ! division below is the first inexact operation.
  x = real(3 + command_argument_count())
  y = 1.0 / x
  print '(A)', "no-trap"
  print *, y
end program

! NOTRAP: no-trap
