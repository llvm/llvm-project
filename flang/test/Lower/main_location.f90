! RUN: split-file %s %t
! RUN: bbc %t/test1.f90 -o - --emit-fir --mlir-print-debuginfo | FileCheck %s --check-prefix=TEST1
! RUN: bbc %t/test2.f90 -o - --emit-fir --mlir-print-debuginfo | FileCheck %s --check-prefix=TEST2
! RUN: bbc %t/test3.f90 -I %t -o - --emit-fir --mlir-print-debuginfo | FileCheck %s --check-prefix=TEST3

! Check that the missing optional program-stmt (R1401)
! does not result in unknown source location of the corresponding
! function.
!
! Also check that the generated `main` entry point is attributed to the start
! of the main program unit rather than to its end-stmt.

!--- test1.f90
if (.false.) then
endif
end

! TEST1: func.func @_QQmain() {
! TEST1-NEXT: fir.dummy_scope : !fir.dscope loc("{{.*}}test1.f90":1:1)
! TEST1-NEXT: return loc("{{.*}}test1.f90":3:1)
! TEST1-NEXT: } loc("{{.*}}test1.f90":1:1)

! TEST1: func.func @main(%{{.*}}: i32 loc("{{.*}}test1.f90":1:1)
! TEST1: } loc("{{.*}}test1.f90":1:1)

!--- test2.f90
!!! keep me here
if (.true.) then
endif
end program

! TEST2: func.func @_QQmain() {
! TEST2-NEXT: fir.dummy_scope : !fir.dscope loc("{{.*}}test2.f90":2:1)
! TEST2-NEXT: return loc("{{.*}}test2.f90":4:1)
! TEST2-NEXT: } loc("{{.*}}test2.f90":2:1)

! TEST2: func.func @main(%{{.*}}: i32 loc("{{.*}}test2.f90":2:1)
! TEST2: } loc("{{.*}}test2.f90":2:1)

!--- inc3.h
integer :: i3
!--- test3.f90
include 'inc3.h'
i3 = 1
end

! When a main program without a program-stmt starts with an INCLUDE, both the
! main program and the generated entry point should get a known location.

! TEST3: func.func @_QQmain() {
! TEST3: } loc(fused<{{.*}}>["{{.*}}inc3.h":1:1, "{{.*}}test3.f90":1:1])

! TEST3: func.func @main(%{{.*}}: i32 loc(fused<{{.*}}>["{{.*}}inc3.h":1:1, "{{.*}}test3.f90":1:1])
! TEST3: fir.call @_QQmain(){{.*}} loc(fused<{{.*}}>["{{.*}}inc3.h":1:1, "{{.*}}test3.f90":1:1])
! TEST3: } loc(fused<{{.*}}>["{{.*}}inc3.h":1:1, "{{.*}}test3.f90":1:1])
