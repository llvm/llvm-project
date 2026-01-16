!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %flang_fc1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=LLVM

!===============================================================================
! Test lowering of prefetch directive
!===============================================================================

subroutine test_prefetch_01()
    ! LLVM: {{.*}} = alloca i32, i64 1, align 4
    ! LLVM: %[[VAR_J:.*]] = alloca i32, i64 1, align 4
    ! LLVM: %[[VAR_I:.*]] = alloca i32, i64 1, align 4
    ! LLVM: %[[VAR_A:.*]] = alloca [256 x i32], i64 1, align 4

    integer :: i, j
    integer :: a(256)

    a = 23
    ! LLVM: call void @llvm.prefetch.p0(ptr %[[VAR_A]], i32 0, i32 3, i32 1)
    !dir$ prefetch a
    i = sum(a)

    ! LLVM: %[[LOAD_I:.*]] = load i32, ptr %[[VAR_I]], align 4
    ! LLVM: %{{.*}} = add nsw i32 %[[LOAD_I]], 64
    ! LLVM: %[[GEP_A:.*]] = getelementptr i32, ptr %[[VAR_A]], i64 {{.*}}

    ! LLVM: call void @llvm.prefetch.p0(ptr %[[GEP_A]], i32 0, i32 3, i32 1)
    ! LLVM: call void @llvm.prefetch.p0(ptr %[[VAR_J]], i32 0, i32 3, i32 1)
    do i = 1, (256 - 64)
      !dir$ prefetch a(i+64), j
      a(i) = a(i-32) + a(i+32) + j
    end do
end subroutine test_prefetch_01
