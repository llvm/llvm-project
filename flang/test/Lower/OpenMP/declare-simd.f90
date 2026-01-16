! This test checks the lowering of the OpenMP `declare simd` directive with
! different clauses (e.g. aligned, linear, simdlen). It also verifies that
! the `omp.declare_simd` operation is emitted after the function prologue,
! since the directive requires access to function arguments.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -cpp -DOMP_60 %s -o - | FileCheck %s

subroutine declare_simd_no_clause()
#ifdef OMP_60
!$omp declare_simd
#else
!$omp declare simd
#endif
end subroutine declare_simd_no_clause

! CHECK-LABEL: func.func @_QPdeclare_simd_no_clause()
! CHECK: omp.declare_simd{{$}}
! CHECK: return

subroutine declare_simd_aligned(x, y, n, i)
#ifdef OMP_60
!$omp declare_simd aligned(x, y : 64)
#else
!$omp declare simd aligned(x, y : 64)
#endif
  real(8), pointer, intent(inout) :: x(:)
  real(8), pointer, intent(in)    :: y(:)
  integer, intent(in) :: n, i
  if (i <= n) x(i) = x(i) + y(i)
end subroutine declare_simd_aligned

! CHECK-LABEL: func.func @_QPdeclare_simd_aligned(
! CHECK: %[[SCOPE_A:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[I_A:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_A]] arg 4 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[N_A:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_A]] arg 3 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[X_A:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_A]] arg 1 {{.*pointer.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK: %[[Y_A:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_A]] arg 2 {{.*pointer.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK: omp.declare_simd aligned(%[[X_A]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>> -> 64 : i64,
! CHECK-SAME:                    %[[Y_A]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>> -> 64 : i64){{$}}
! CHECK: return

subroutine declare_simd_linear(x, y, n, i)
#ifdef OMP_60
!$omp declare_simd linear(i)
#else
!$omp declare simd linear(i)
#endif
  real(8), pointer, intent(inout) :: x(:)
  real(8), pointer, intent(in)    :: y(:)
  integer, intent(in) :: n, i
  if (i <= n) x(i) = x(i) + y(i)
end subroutine  declare_simd_linear

! CHECK-LABEL: func.func @_QPdeclare_simd_linear(
! CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[I:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE]] arg 4 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[C1:.*]] = arith.constant 1 : i32
! CHECK: omp.declare_simd linear(%[[I]]#0 = %[[C1]] : !fir.ref<i32>) {linear_var_types = [i32]}{{$}}
! CHECK: return

subroutine declare_simd_simdlen(x, y, n, i)
#ifdef OMP_60
!$omp declare_simd simdlen(8)
#else
!$omp declare simd simdlen(8)
#endif
end subroutine declare_simd_simdlen

! CHECK-LABEL: func.func @_QPdeclare_simd_simdlen(
! CHECK: %[[SCOPE_S:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: omp.declare_simd{{.*}}simdlen(8){{$}}
! CHECK-NEXT: return

subroutine declare_simd_combined(x, y, n, i)
#ifdef OMP_60
!$omp declare_simd aligned(x, y : 64) linear(i) simdlen(8)
#else
!$omp declare simd aligned(x, y : 64) linear(i) simdlen(8)
#endif
  real(8), pointer, intent(inout) :: x(:)
  real(8), pointer, intent(in)    :: y(:)
  integer, intent(in) :: n, i

  if (i <= n) then
      x(i) = x(i) + y(i)
  end if
end subroutine declare_simd_combined

! CHECK-LABEL: func.func @_QPdeclare_simd_combined(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>{{.*}}fir.bindc_name = "x"
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>{{.*}}fir.bindc_name = "y"
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}fir.bindc_name = "n"
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}fir.bindc_name = "i"
! CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE]] arg 4 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[N_DECL:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE]] arg 3 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE]] arg 1 {{.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE]] arg 2 {{.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK: %[[C1:.*]] = arith.constant 1 : i32

! CHECK: omp.declare_simd
! CHECK-SAME: aligned(%[[X_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>> -> 64 : i64,
! CHECK-SAME:         %[[Y_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>> -> 64 : i64)
! CHECK-SAME: linear(%[[I_DECL]]#0 = %[[C1]] : !fir.ref<i32>)
! CHECK-SAME: simdlen(8)
! CHECK-SAME: {linear_var_types = [i32]}
