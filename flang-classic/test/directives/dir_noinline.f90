!! check for pragma support for no inlining of functions
!RUN: %flang -S -emit-llvm %s -o - | FileCheck %s
!RUN: %flang -O2 -S -emit-llvm %s -o - | FileCheck %s
!RUN: %flang -O3 -S -emit-llvm %s -o - | FileCheck %s

!CHECK: define void @func_noinline_(){{.*}} #0 {{.*$}}
!CHECK: call void @func_noinline_(), {{.*$}}
!CHECK: attributes #0 = { noinline {{.*$}}

!DIR$ NOINLINE
SUBROUTINE func_noinline
    INTEGER :: i
    do i = 0, 5
            WRITE(*, *) "Hello World"
    end do
END SUBROUTINE func_noinline

PROGRAM test_inline
    IMPLICIT NONE
    call func_noinline
END PROGRAM test_inline
