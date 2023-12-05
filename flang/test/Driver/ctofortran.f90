! MacOS needs -isysroot <osx_sysroot> with clang and flang to build binaries.
! UNSUPPORTED: system-windows, system-darwin
! RUN: split-file %s %t
! RUN: chmod +x %t/runtest.sh
! RUN: %t/runtest.sh %t %flang $t/ffile.f90 $t/cfile.c

!--- ffile.f90
program fmain
  interface
    subroutine csub() bind(c)
    end subroutine
  end interface

  call csub()
end program fmain

subroutine foo(a) bind(c)
  integer :: a(:)
  if (lbound(a, 1) .ne. 1) then
     print *, 'FAIL expected 1 for lbound but got ',lbound(a, 1)
     stop 1
  endif

  if (ubound(a, 1) .ne. 10) then
     print *, 'FAIL expected 10 for ubound but got ',ubound(a, 1)
     stop 1
  endif

  do i = lbound(a,1),ubound(a,1)
     !print *, a(i)
     if (a(i) .ne. i) then
        print *, 'FAIL expected', i, ' for index ',i, ' but got ',a(i)
        stop 1
     endif
  enddo
  print *, 'PASS'
end subroutine foo

! CHECK: PASS
!--- cfile.c
#include <stdio.h>
#include <stdlib.h>
#include <ISO_Fortran_binding.h>

void foo(CFI_cdesc_t*);

int a[10];

void csub() {
  int i, res;
  static CFI_CDESC_T(1) r1;
  CFI_cdesc_t *desc = (CFI_cdesc_t*)&r1;
  CFI_index_t extent[1] = {10};

  for(i=0; i<10; ++i) {
    a[i] = i+1;
  }

  res = CFI_establish(desc, (void*)a, CFI_attribute_other, CFI_type_int32_t,
                      sizeof(int), 1, extent);
  if (res != 0) {
    printf("FAIL CFI_establish returned %d instead of 0.\n",res);
    exit(1);
  }

  foo(desc);
  return;
}
!--- runtest.sh
#!/bin/bash
export BINDIR=`dirname $2`
export CCOMP=$BINDIR/clang
if [ -x $CCOMP ]
then
  export LIBDIR=$BINDIR/../lib
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBDIR
  $CCOMP -c $1/$4 -o $1/cfile.o
  $2 $1/$3 $1/cfile.o -o $1/ctofortran
  $1/ctofortran # should print "PASS"
else
  # No clang compiler, just pass by default
  echo "PASS"
fi
