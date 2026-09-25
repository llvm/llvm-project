! Test that -ffpe-trap=underflow enables run-time halting on an IEEE_UNDERFLOW
! exception.

! Halting requires both glibc (feenableexcept) and hardware that delivers a
! trap when an enabled FP exception is raised. Arm makes FP-exception trapping
! optional, and the AArch64 CI hardware does not deliver SIGFPE, so restrict
! this run-time test to x86 glibc.
! REQUIRES: target=x86_64{{.*}}-linux-gnu
! UNSUPPORTED: offload-cuda

! Built without traps: the underflowing multiplication completes and the
! program exits normally.
! RUN: %flang %isysroot -L"%libdir" %s -o %t.notrap
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t.notrap \
! RUN:     | FileCheck --check-prefix=NOTRAP %s

! Built with -ffpe-trap=underflow: squaring the smallest normal value yields a
! result below the subnormal range, raising IEEE_UNDERFLOW, and halting
! terminates the program with SIGFPE (signal 8, i.e. shell exit 136).
! RUN: %flang %isysroot -L"%libdir" -ffpe-trap=underflow %s -o %t.trap
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" sh -c 'ulimit -c 0; %t.trap; test $? -eq 136'

program fpe_trap_underflow
  real :: x, y
  ! x == tiny (the smallest normal), computed via a runtime-unknown factor so
  ! the multiplication below is not folded. tiny*1.0 is exact (no underflow).
  x = tiny(0.0) * real(command_argument_count() + 1)
  y = x * x
  print '(A)', "no-trap"
  print *, y
end program

! NOTRAP: no-trap
