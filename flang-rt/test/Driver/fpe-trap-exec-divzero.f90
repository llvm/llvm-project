! Test that -ffpe-trap=zero enables run-time halting on an IEEE_DIVIDE_BY_ZERO
! exception, and that trapping is selective (a different enabled trap does not
! halt on a divide-by-zero).

! Halting requires both glibc (feenableexcept) and hardware that delivers a
! trap when an enabled FP exception is raised. Arm makes FP-exception trapping
! optional, and the AArch64 CI hardware does not deliver SIGFPE, so restrict
! this run-time test to x86 glibc.
! REQUIRES: target=x86_64{{.*}}-linux-gnu
! UNSUPPORTED: offload-cuda

! Built without traps: the division by zero yields infinity and the program
! exits normally.
! RUN: %flang %isysroot -L"%libdir" %s -o %t.notrap
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t.notrap \
! RUN:     | FileCheck --check-prefix=NOTRAP %s

! Built with -ffpe-trap=zero: the division by zero raises IEEE_DIVIDE_BY_ZERO
! and halting terminates the program with SIGFPE (signal 8, i.e. shell exit 136).
! RUN: %flang %isysroot -L"%libdir" -ffpe-trap=zero %s -o %t.trap
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" sh -c 'ulimit -c 0; %t.trap; test $? -eq 136'

! Selectivity: trapping on overflow only must not halt on a divide-by-zero.
! RUN: %flang %isysroot -L"%libdir" -ffpe-trap=overflow %s -o %t.other
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t.other \
! RUN:     | FileCheck --check-prefix=NOTRAP %s

program fpe_trap_divzero
  real :: x, y
  ! command_argument_count() is not known at compile time, which prevents the
  ! division from being folded away by the compiler.
  x = real(command_argument_count())
  y = 1.0 / x
  print '(A)', "no-trap"
  print *, y
end program

! NOTRAP: no-trap
