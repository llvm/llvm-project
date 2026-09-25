! Test that -ffpe-trap= actually enables halting at run time: an invalid
! floating-point operation must terminate the program with a signal (SIGFPE)
! when the corresponding trap is enabled, and must complete normally otherwise.

! Halting requires both glibc (feenableexcept) and hardware that delivers a
! trap when an enabled FP exception is raised. Arm makes FP-exception trapping
! optional, and the AArch64 CI hardware does not deliver SIGFPE, so restrict
! this run-time test to x86 glibc.
! REQUIRES: target=x86_64{{.*}}-linux-gnu
! UNSUPPORTED: offload-cuda

! Built without traps: the invalid operation yields a NaN and the program exits
! normally.
! RUN: %flang %isysroot -L"%libdir" %s -o %t.notrap
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t.notrap \
! RUN:     | FileCheck --check-prefix=NOTRAP %s

! Built with -ffpe-trap=invalid: the invalid operation raises IEEE_INVALID and
! halting terminates the program with SIGFPE (signal 8, i.e. shell exit 136).
! RUN: %flang %isysroot -L"%libdir" -ffpe-trap=invalid %s -o %t.trap
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" sh -c 'ulimit -c 0; %t.trap; test $? -eq 136'

program fpe_trap_exec
  real :: x, y
  ! command_argument_count() is not known at compile time, which prevents the
  ! invalid operation from being folded away by the compiler.
  x = real(command_argument_count()) - 1.0
  y = sqrt(x)
  print '(A)', "no-trap"
  print *, y
end program

! NOTRAP: no-trap
