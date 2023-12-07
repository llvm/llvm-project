! RUN: split-file %s %t
! RUN: bbc %t/test1.f90 -o - --emit-fir --mlir-print-debuginfo | FileCheck %s --check-prefix=TEST1
! RUN: bbc %t/test2.f90 -o - --emit-fir --mlir-print-debuginfo | FileCheck %s --check-prefix=TEST2

! Check that the missing optional program-stmt (R1401)
! does not result in unknown source location of the corresponding
! function.

!--- test1.f90
if (.false.) then
endif
end

! TEST1: func.func @_QQmain() {
! TEST1-NEXT: return loc("{{.*}}test1.f90":3:1)
! TEST1-NEXT: } loc("{{.*}}test1.f90":1:1)

!--- test2.f90
!!! keep me here
if (.true.) then
endif
end program

! TEST2: func.func @_QQmain() {
! TEST2-NEXT: return loc("{{.*}}test2.f90":4:1)
! TEST2-NEXT: } loc("{{.*}}test2.f90":2:1)
