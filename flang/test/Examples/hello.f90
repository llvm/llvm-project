! Note: On linux, the Fortran runtime wants to include libm as well.
! Note: Add pic to the compilation as gcc will use shared libraries by
! default when they are available.

! RUN: bbc %s -o - | tco | llc --relocation-model=pic --filetype=obj -o %t.o
! RUN: %CC %t.o -L%L -Wl,-rpath -Wl,%L -lFortran_main -lFortranRuntime -lFortranDecimal -lm -o hello
! RUN: ./hello | FileCheck %s

! CHECK: Hello, World!
  print *, "Hello, World!"
  end
