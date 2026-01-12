! This test checks the lowering of the OpenMP `declare simd` directive with
! different clauses (e.g. aligned, linear, simdlen). It also verifies that
! the `omp.declare_simd` operation is emitted after the function prologue,
! since the directive requires access to function arguments.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine declare_simd_no_clause(x, y, n, i)
  !$omp declare simd
  real(8), pointer, intent(inout) :: x(:)
  real(8), pointer, intent(in)    :: y(:)
  integer, intent(in) :: n, i
  if (i <= n) x(i) = x(i) + y(i)
end subroutine declare_simd_no_clause

! CHECK-LABEL: func.func @_QPdeclare_simd_no_clause(
! CHECK: %[[SCOPE_NONE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[I_NONE:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_NONE]] arg 4 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[N_NONE:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_NONE]] arg 3 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[X_NONE:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_NONE]] arg 1 {{.*pointer.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK: %[[Y_NONE:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_NONE]] arg 2 {{.*pointer.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK: omp.declare_simd
! CHECK-NOT: aligned(
! CHECK-NOT: linear(
! CHECK-NOT: simdlen(
! CHECK: return

subroutine declare_simd_aligned(x, y, n, i)
  !$omp declare simd aligned(x, y : 64)
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
! CHECK: omp.declare_simd
! CHECK-SAME: aligned(%[[X_A]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>> -> 64 : i64,
! CHECK-SAME:         %[[Y_A]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>> -> 64 : i64)
! CHECK-NOT: linear(
! CHECK-NOT: simdlen(
! CHECK: return

subroutine declare_simd_linear(x, y, n, i)
  !$omp declare simd linear(i)
  real(8), pointer, intent(inout) :: x(:)
  real(8), pointer, intent(in)    :: y(:)
  integer, intent(in) :: n, i
  if (i <= n) x(i) = x(i) + y(i)
end subroutine  declare_simd_linear

! CHECK-LABEL: func.func @_QPdeclare_simd_linear(
! CHECK: %[[SCOPE_L:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[I_L:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_L]] arg 4 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[N_L:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_L]] arg 3 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[X_L:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_L]] arg 1 {{.*pointer.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK: %[[Y_L:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_L]] arg 2 {{.*pointer.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK: %[[C1_L:.*]] = arith.constant 1 : i32
! CHECK: omp.declare_simd
! CHECK-NOT: aligned(
! CHECK-SAME: linear(%[[I_L]]#0 = %[[C1_L]] : !fir.ref<i32>)
! CHECK-SAME: {linear_var_types = [i32]}
! CHECK-NOT: simdlen(
! CHECK: return

subroutine declare_simd_simdlen(x, y, n, i)
  !$omp declare simd simdlen(8)
  real(8), pointer, intent(inout) :: x(:)
  real(8), pointer, intent(in)    :: y(:)
  integer, intent(in) :: n, i
  if (i <= n) x(i) = x(i) + y(i)
end subroutine declare_simd_simdlen

! CHECK-LABEL: func.func @_QPdeclare_simd_simdlen(
! CHECK: %[[SCOPE_S:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[I_S:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_S]] arg 4 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[N_S:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_S]] arg 3 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[X_S:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_S]] arg 1 {{.*pointer.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK: %[[Y_S:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_S]] arg 2 {{.*pointer.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK: omp.declare_simd
! CHECK-NOT: aligned(
! CHECK-NOT: linear(
! CHECK-SAME: simdlen(8)
! CHECK: return

subroutine declare_simd_uniform(x, y, n, i)
  !$omp declare simd uniform(x, y)

  real(8), pointer, intent(inout) :: x(:)
  real(8), pointer, intent(in)    :: y(:)
  integer, intent(in) :: n, i

  if (i <= n) then
    x(i) = x(i) + y(i)
  end if
end subroutine declare_simd_uniform

! CHECK-LABEL: func.func @_QPdeclare_simd_uniform(
! CHECK-SAME: fir.bindc_name = "x"
! CHECK-SAME: fir.bindc_name = "y"
! CHECK-SAME: fir.bindc_name = "n"
! CHECK-SAME: fir.bindc_name = "i"

! Function prologue
! CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[IDECL:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE]] arg 4
! CHECK: %[[NDECL:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE]] arg 3
! CHECK: %[[XDECL:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE]] arg 1
! CHECK: %[[YDECL:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE]] arg 2

! CHECK: omp.declare_simd uniform(%[[XDECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, %[[YDECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)

subroutine declare_simd_combined(x, y, n, i)
    !$omp declare simd aligned(x, y : 64) linear(i) simdlen(8) uniform(x, y)
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
! CHECK-SAME: uniform(%[[X_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>,
! CHECK-SAME:         %[[Y_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK-SAME: {linear_var_types = [i32]}
