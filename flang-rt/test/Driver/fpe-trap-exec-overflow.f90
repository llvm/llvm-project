! Test that -ffpe-trap=overflow enables run-time halting on an IEEE_OVERFLOW
! exception.

! Halting requires both glibc (feenableexcept) and hardware that delivers a
! trap when an enabled FP exception is raised. Arm makes FP-exception trapping
! optional, and the AArch64 CI hardware does not deliver SIGFPE, so restrict
! this run-time test to x86 glibc.
! REQUIRES: target=x86_64{{.*}}-linux-gnu
! UNSUPPORTED: offload-cuda

! Built without traps: the overflowing multiplication yields infinity and the
! program exits normally.
! RUN: %flang %isysroot -L"%libdir" %s -o %t.notrap
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t.notrap \
! RUN:     | FileCheck --check-prefix=NOTRAP %s

! Built with -ffpe-trap=overflow: the multiplication raises IEEE_OVERFLOW and
! halting terminates the program with SIGFPE (signal 8, i.e. shell exit 136).
! RUN: %flang %isysroot -L"%libdir" -ffpe-trap=overflow %s -o %t.trap
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" sh -c 'ulimit -c 0; %t.trap; test $? -eq 136'

program fpe_trap_overflow
  real :: x, y
  ! Subtracting a runtime-unknown value keeps x == huge while preventing the
  ! compiler from folding the overflowing multiplication.
  x = huge(0.0) - real(command_argument_count())
  y = x * x
  print '(A)', "no-trap"
  print *, y
end program

! NOTRAP: no-trap
