! A single-type user reduction whose name happens to end in a type token
! (myred_i32) must not collide with a multi-type reduction (myred: integer, real)
! whose per-type integer op is also spelled with an _i32 suffix. Because the
! per-type suffix is appended unconditionally, the single-type reduction becomes
! @_QQFmyred_i32_i32, distinct from the multi-type reduction's @_QQFmyred_i32, so
! the by-name op dedup cannot silently fold one reduction onto the other. A
! conditional suffix (bare single-type name) would collide here and silently bind
! myred_i32's clause to myred's combiner (a miscompile).
! See https://github.com/llvm/llvm-project/issues/207255.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

program p
  integer :: a, b, i
  !$omp declare reduction(myred: integer, real : omp_out = omp_out + omp_in) &
  !$omp   initializer(omp_priv = 0)
  !$omp declare reduction(myred_i32: integer : omp_out = omp_out * omp_in) &
  !$omp   initializer(omp_priv = 1)
  a = 0
  b = 1
  !$omp parallel do reduction(myred:a)
  do i = 1, 4
    a = a + i
  end do
  !$omp parallel do reduction(myred_i32:b)
  do i = 1, 4
    b = b * i
  end do
  print *, a, b
end program

! Three distinct ops: the multi-type reduction's two per-type ops, and the
! single-type reduction's own (double-suffixed) op. Under a conditional suffix
! only two ops would exist and the single-type reduction would silently reuse
! the multi-type add op; three distinct ops here proves there is no collision.
! CHECK-DAG: omp.declare_reduction @[[MYRED_I32:_QQFmyred_i32]] : i32
! CHECK-DAG: omp.declare_reduction @[[MYRED_F32:_QQFmyred_f32]] : f32
! CHECK-DAG: omp.declare_reduction @[[MYREDI32_I32:_QQFmyred_i32_i32]] : i32

! Each loop binds its own reduction's op, by full name. Loop 2 (myred_i32) binds
! its own @_QQFmyred_i32_i32, not the multi-type reduction's @_QQFmyred_i32.
! CHECK-DAG: reduction(@[[MYRED_I32]] %{{[^)]*}}!fir.ref<i32>
! CHECK-DAG: reduction(@[[MYREDI32_I32]] %{{[^)]*}}!fir.ref<i32>
