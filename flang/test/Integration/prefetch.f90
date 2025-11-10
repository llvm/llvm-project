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
    ! LLVM: %[[L_J:.*]] = alloca i32, i64 1, align 4
    ! LLVM: %[[L_I:.*]] = alloca i32, i64 1, align 4
    ! LLVM: %[[L_A:.*]] = alloca [256 x i32], i64 1, align 4

    integer :: i, j
    integer :: a(256)

    a = 23
    ! LLVM: call void @llvm.prefetch.p0(ptr %6, i32 0, i32 3, i32 1)
    !dir$ prefetch a
    i = sum(a)
    ! LLVM: %[[L_LOAD:.*]] = load i32, ptr %5, align 4
    ! LLVM: %[[L_ADD:.*]] = add nsw i32 %[[L_LOAD]], 64
    ! LLVM: %[[L_GEP:.*]] = getelementptr i32, ptr %[[L_A]], i64 {{.*}}

    ! LLVM: call void @llvm.prefetch.p0(ptr %[[L_GEP]], i32 0, i32 3, i32 1)
    ! LLVM: call void @llvm.prefetch.p0(ptr %[[L_J]], i32 0, i32 3, i32 1)

    do i = 1, (256 - 64)
      !dir$ prefetch a(i+64), j
      a(i) = a(i-32) + a(i+32) + j
    end do
end subroutine test_prefetch_01
