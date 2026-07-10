! When compiling with debug info, the "file name" should be "<stdin>" for
! consistency with clang and gfortran. The exception is the first file that is
! referenced from the DICompileUnit in which the file name is "-". This is also
! consistent with clang.
!
! RUN: cat %s \
! RUN:     | %flang -g -x f95 -S -emit-llvm -o - - \
! RUN:     | FileCheck %s
!
! CHECK: !DICompileUnit({{.*}}file: ![[DASH:[0-9]+]]
! CHECK: ![[DASH]] = !DIFile({{.*}}filename: "-"
! CHECK: !DIFile({{.*}}filename: "<stdin>"

end program
