! Test that -ffpe-trap=denormal enables run-time halting on the (non-standard,
! gfortran-compatible) denormal-operand exception.

! The denormal-operand exception is an x86 SSE feature (__FE_DENORM); on other
! architectures it is not available, so restrict this test to x86 glibc.
! REQUIRES: target=x86_64{{.*}}-linux-gnu
! UNSUPPORTED: offload-cuda

! Built without traps: the operation on a subnormal operand completes and the
! program exits normally.
! RUN: %flang %isysroot -L"%libdir" %s -o %t.notrap
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t.notrap \
! RUN:     | FileCheck --check-prefix=NOTRAP %s

! Built with -ffpe-trap=denormal: adding a subnormal operand raises the
! denormal-operand exception and halting terminates the program with SIGFPE
! (signal 8, i.e. shell exit 136).
! RUN: %flang %isysroot -L"%libdir" -ffpe-trap=denormal %s -o %t.trap
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" sh -c 'ulimit -c 0; %t.trap; test $? -eq 136'

program fpe_trap_denormal
  real :: x, y
  ! Build a subnormal value at runtime. Its operands are normal, so this does
  ! not itself raise the denormal-operand exception, and 2**-127 is an exact
  ! subnormal, so it does not raise underflow either.
  x = tiny(0.0) * (0.5 + real(command_argument_count()))
  ! x is subnormal, so this operation has a denormal operand.
  y = x + x
  print '(A)', "no-trap"
  print *, y
end program

! NOTRAP: no-trap
