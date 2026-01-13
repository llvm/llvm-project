! This test CHECKs the lowering of the OpenMP `declare simd` directive with
! different clauses (e.g. aligned, linear, simdlen). It also verifies that
! the `omp.declare_simd` operation is emitted after the function prologue,
! since the directive requires access to function arguments.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -cpp -DOMP_60 %s -o - | FileCheck %s --check-prefix=CHECK60

subroutine declare_simd_no_clause()
  !$omp declare simd
end subroutine declare_simd_no_clause

! CHECK-LABEL: func.func @_QPdeclare_simd_no_clause()
! CHECK: omp.declare_simd
! CHECK-NOT: {{omp\.declare_simd[[:space:]]*(aligned|linear|simdlen)\(}}
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
! CHECK-NOT: {{[[:space:]]*(linear|simdlen)\(}}
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
! CHECK-NOT: {{[[:space:]]*(aligned|simdlen)\(}}
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
! CHECK-NOT: {{[[:space:]]*(aligned|linear)\(}}
! CHECK-SAME: simdlen(8)
! CHECK: return

subroutine declare_simd_combined(x, y, n, i)
    !$omp declare simd aligned(x, y : 64) linear(i) simdlen(8)
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

#ifdef OMP_60

subroutine declare_simd_60_no_clause()
  !$omp declare_simd
end subroutine declare_simd_60_no_clause

! CHECK60-LABEL: func.func @_QPdeclare_simd_60_no_clause() {
! CHECK60: omp.declare_simd

subroutine declare_simd_aligned_60(x, y, n, i)
  !$omp declare_simd aligned(x, y : 64)
  real(8), pointer, intent(inout) :: x(:)
  real(8), pointer, intent(in)    :: y(:)
  integer, intent(in) :: n, i
  if (i <= n) x(i) = x(i) + y(i)
end subroutine declare_simd_aligned_60

! CHECK60-LABEL: func.func @_QPdeclare_simd_aligned_60(
! CHECK60: %[[SCOPE_A:.*]] = fir.dummy_scope : !fir.dscope
! CHECK60: %[[I_A:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_A]] arg 4 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK60: %[[N_A:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_A]] arg 3 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK60: %[[X_A:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_A]] arg 1 {{.*pointer.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK60: %[[Y_A:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_A]] arg 2 {{.*pointer.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK60: omp.declare_simd
! CHECK60-SAME: aligned(%[[X_A]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>> -> 64 : i64,
! CHECK60-SAME:         %[[Y_A]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>> -> 64 : i64)
! CHECK60-NOT: {{[[:space:]]*(linear|simdlen)\(}}
! CHECK60: return

subroutine declare_simd_linear_60(x, y, n, i)
  !$omp declare_simd linear(i)
  real(8), pointer, intent(inout) :: x(:)
  real(8), pointer, intent(in)    :: y(:)
  integer, intent(in) :: n, i
  if (i <= n) x(i) = x(i) + y(i)
end subroutine  declare_simd_linear_60

! CHECK60-LABEL: func.func @_QPdeclare_simd_linear_60(
! CHECK60: %[[SCOPE_L:.*]] = fir.dummy_scope : !fir.dscope
! CHECK60: %[[I_L:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_L]] arg 4 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK60: %[[N_L:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_L]] arg 3 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK60: %[[X_L:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_L]] arg 1 {{.*pointer.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK60: %[[Y_L:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_L]] arg 2 {{.*pointer.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK60: %[[C1_L:.*]] = arith.constant 1 : i32
! CHECK60: omp.declare_simd
! CHECK60-NOT: {{[[:space:]]*(aligned|simdlen)\(}}
! CHECK60: return

subroutine declare_simd_simdlen_60(x, y, n, i)
  !$omp declare_simd simdlen(8)
  real(8), pointer, intent(inout) :: x(:)
  real(8), pointer, intent(in)    :: y(:)
  integer, intent(in) :: n, i
  if (i <= n) x(i) = x(i) + y(i)
end subroutine declare_simd_simdlen_60

! CHECK60-LABEL: func.func @_QPdeclare_simd_simdlen_60(
! CHECK60: %[[SCOPE_S:.*]] = fir.dummy_scope : !fir.dscope
! CHECK60: %[[I_S:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_S]] arg 4 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK60: %[[N_S:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_S]] arg 3 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK60: %[[X_S:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_S]] arg 1 {{.*pointer.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK60: %[[Y_S:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE_S]] arg 2 {{.*pointer.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK60: omp.declare_simd
! CHECK60-NOT: {{[[:space:]]*(aligned|linear)\(}}
! CHECK60-SAME: simdlen(8)
! CHECK60: return

subroutine declare_simd_combined_60(x, y, n, i)
    !$omp declare_simd aligned(x, y : 64) linear(i) simdlen(8)
    real(8), pointer, intent(inout) :: x(:)
    real(8), pointer, intent(in)    :: y(:)
    integer, intent(in) :: n, i

    if (i <= n) then
        x(i) = x(i) + y(i)
    end if
end subroutine declare_simd_combined_60

! CHECK60-LABEL: func.func @_QPdeclare_simd_combined_60(
! CHECK60-SAME: %{{.*}}: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>{{.*}}fir.bindc_name = "x"
! CHECK60-SAME: %{{.*}}: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>{{.*}}fir.bindc_name = "y"
! CHECK60-SAME: %{{.*}}: !fir.ref<i32>{{.*}}fir.bindc_name = "n"
! CHECK60-SAME: %{{.*}}: !fir.ref<i32>{{.*}}fir.bindc_name = "i"
! CHECK60: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK60: %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE]] arg 4 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK60: %[[N_DECL:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE]] arg 3 {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK60: %[[X_DECL:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE]] arg 1 {{.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK60: %[[Y_DECL:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[SCOPE]] arg 2 {{.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
! CHECK60: %[[C1:.*]] = arith.constant 1 : i32

! CHECK60: omp.declare_simd
! CHECK60-SAME: aligned(%[[X_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>> -> 64 : i64,
! CHECK60-SAME:         %[[Y_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>> -> 64 : i64)
! CHECK60-SAME: linear(%[[I_DECL]]#0 = %[[C1]] : !fir.ref<i32>)
! CHECK60-SAME: simdlen(8)
! CHECK60-SAME: {linear_var_types = [i32]}

#endif !OMP_60
