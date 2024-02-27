! If the compiler is built without 128-bit float math
! support, an appropriate error message is emitted.
! UNSUPPORTED: flang-supports-f128-math
! RUN: bbc -emit-fir %s -o /dev/null >%t 2>&1 || echo
! RUN: FileCheck %s --input-file=%t

 complex(16) :: a
 real(16) :: b
! CHECK: compiler is built without support for 'ABS(COMPLEX(KIND=16))'
 b = abs(a)
end

