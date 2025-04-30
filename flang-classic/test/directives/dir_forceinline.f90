!! check for pragma support for forced inlining of functions
!RUN: %flang -S -emit-llvm %s -o - | FileCheck %s
!RUN: %flang -O2 -S -emit-llvm %s -o - | FileCheck %s
!RUN: %flang -O3 -S -emit-llvm %s -o - | FileCheck %s

!CHECK: define void @func_forceinline_(){{.*}} #0 {{.*$}}
!CHECK-NOT: call void @func_forceinline_(), {{.*$}}
!CHECK: attributes #0 = { alwaysinline {{.*$}}

!DIR$ FORCEINLINE
SUBROUTINE func_forceinline
    INTEGER :: i
    do i = 0, 5
            WRITE(*, *) "Hello World"
    end do
END SUBROUTINE func_forceinline

PROGRAM test_inline
    IMPLICIT NONE
    call func_forceinline
END PROGRAM test_inline
