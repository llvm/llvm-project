! This directory can be used to add Integration tests involving multiple stages of the compiler (for eg. from Fortran to LLVM IR).
! It should not contain executable tests. We should only add tests here sparingly and only if there is no other way to test.
! RUN: %flang_fc1 -emit-llvm -o - %s | FileCheck %s

! CHECK-LABEL: test_inline
subroutine test_inline()
  integer :: x, y
!CHECK:  %[[VAL_1:.*]] = alloca i32, i64 1, align 4
!CHECK:  %[[VAL_2:.*]] = alloca i32, i64 1, align 4
!CHECK:  %[[VAL_3:.*]] = alloca i32, i64 1, align 4
!CHECK:  %[[VAL_4:.*]] = alloca i32, i64 1, align 4

  !dir$ forceinline
  y = g(x)
  !dir$ forceinline
  call f(x, y)
!CHECK:  %[[VAL_5:.*]] = load i32, ptr %[[VAL_3]], align 4
!CHECK:  %[[VAL_6:.*]] = mul i32 %[[VAL_5]], 2
!CHECK:  store i32 %6, ptr %[[VAL_1]], align 4
!CHECK:  %[[VAL_7:.*]] = load i32, ptr %[[VAL_1]], align 4
!CHECK:  store i32 %7, ptr %[[VAL_2]], align 4
!CHECK:  %[[VAL_8:.]] = load i32, ptr %[[VAL_3]], align 4
!CHECK:  %[[VAL_9:.]] = mul i32 %[[VAL_8]], 2
!CHECK:  store i32 %9, ptr %[[VAL_2]], align 4

  !dir$ inline
  y = g(x)
  !dir$ inline
  call f(x, y)
!CHECK:  %[[VAL_10:.*]] = call i32 @_QFtest_inlinePg(ptr %[[VAL_3]]) #[[INLINE:.*]]
!CHECK:  store i32 %[[VAL_10]], ptr %[[VAL_2]], align 4
!CHECK:  call void @_QFtest_inlinePf(ptr %[[VAL_3]], ptr %[[VAL_2]]) #[[INLINE]]

  !dir$ inline
  do i = 1, 100
    call f(x, y)
    !CHECK: br i1 %[[VAL_14:.*]], label %[[VAL_15:.*]], label %[[VAL_19:.*]]
    !CHECK: call void @_QFtest_inlinePf(ptr %[[VAL_3]], ptr %[[VAL_2]]) #[[INLINE]]
  enddo

  !dir$ noinline
  y = g(x)
  !dir$ noinline
  call f(x, y)
!CHECK:  %[[VAL_10:.*]] = call i32 @_QFtest_inlinePg(ptr %[[VAL_3]]) #[[NOINLINE:.*]]
!CHECK:  store i32 %[[VAL_10]], ptr %[[VAL_2]], align 4
!CHECK:  call void @_QFtest_inlinePf(ptr %[[VAL_3]], ptr %[[VAL_2]]) #[[NOINLINE]]

  !dir$ noinline
  do i = 1, 100
    call f(x, y)
    !CHECK: br i1 %[[VAL_14:.*]], label %[[VAL_15:.*]], label %[[VAL_19:.*]]
    !CHECK: call void @_QFtest_inlinePf(ptr %[[VAL_3]], ptr %[[VAL_2]]) #[[NOINLINE]]
  enddo

  contains
    subroutine f(x, y)
      integer, intent(in) :: x
      integer, intent(out) :: y
      y = x*2
    end subroutine f
    integer function g(x)
      integer :: x
      g = x*2
    end function g
end subroutine test_inline

!CHECK: attributes #[[INLINE]] = { inlinehint }
!CHECK: attributes #[[NOINLINE]] = { noinline }
