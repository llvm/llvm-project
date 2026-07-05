! RUN: %flang -E %s 2>&1 | FileCheck --strict-whitespace %s
!CHECK:       program Main
program Main
#define ADD(x,y) (x)+(y)
!CHECK:       integer :: j = (1)+( 2)
  integer :: j = ADD(1,&
                     2)
!CHECK:1     format('This is a very long output literal edit descriptor for a F&
!CHECK:     &ORMAT statement, and it will require statement continuation.')
1 format('This is a very long output literal edit descriptor for a FORMAT statement, and it will require statement continuation.')












!CHECK: #line 25
!CHECK:       end PROGRAM Main
end PROGRAM Main
