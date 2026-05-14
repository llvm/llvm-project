! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - 2>&1 | FileCheck %s --check-prefixes=CHECK,DEFAULT
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - 2>&1 | FileCheck %s --check-prefixes=CHECK,OPENMP52


subroutine do_simd
!CHECK: %[[I:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFdo_simdEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[X:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFdo_simdEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[CONST:.*]] = arith.constant 1 : i32
!CHECK: %{{.*}} = arith.constant 1 : i32
!CHECK: %{{.*}} = arith.constant 1 : i32
!DEFAULT: omp.wsloop linear(%[[X]]#0 : !fir.ref<i32> = {{.*}}) {
!OPENMP52: omp.wsloop linear(val(%[[X]]#0 : !fir.ref<i32> = {{.*}})) {
!CHECK: omp.simd linear({{.*}}) private(@_QFdo_simdEi_private_i32 {{.*}} -> %arg0 : !fir.ref<i32>) {
!CHECK: }
!CHECK: } {linear_var_types = [i32], omp.composite}
!CHECK: } {linear_var_types = [i32], omp.composite}
    integer :: x
    !$omp do simd linear(x:1)
    do i = 1, N
    end do
    !$omp end do simd
end subroutine do_simd


subroutine distribute_simd
!CHECK: omp.teams {
!CHECK: omp.distribute private(@_QFdistribute_simdEi_private_i32 {{.*}} -> %[[ARG0:.*]] : !fir.ref<i32>) {
!DEFAULT: omp.simd linear({{.*}}) {
!OPENMP52: omp.simd linear(val({{.*}})) private(@_QFdistribute_simdEi_private_i32 %[[ARG0]] -> {{.*}} : !fir.ref<i32>) {
!CHECK: } {linear_var_types = [i32], omp.composite}
!CHECK: } {omp.composite}
    integer :: i
    integer :: x
    !$omp teams
    !$omp distribute simd linear(i:1)
    do i = 1, N
    end do
    !$omp end distribute simd
    !$omp end teams
end subroutine distribute_simd


subroutine distribute_parallel_do
!CHECK: %[[I:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFdistribute_parallel_doEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.teams {
!CHECK: omp.parallel {
!CHECK: %[[CONST]] = arith.constant 1 : i32
!CHECK: %{{.*}} = arith.constant 1 : i32
!CHECK: %[[IV_STEP:.*]] = arith.constant 1 : i32
!CHECK: omp.distribute {
!DEFAULT: omp.wsloop linear(%[[I]]#0 : !fir.ref<i32> = {{.*}}) {
!OPENMP52: omp.wsloop linear(val(%[[I]]#0 : !fir.ref<i32> = {{.*}})) {
!CHECK: omp.simd linear({{.*}}) private(@_QFdistribute_parallel_doEi_private_i32 {{.*}}) {
    !$omp teams
    !$omp distribute parallel do simd linear(i:1)
    do i = 1, N
    end do
    !$omp end distribute parallel do simd
!CHECK: } {linear_var_types = [i32], omp.composite}
!CHECK: } {linear_var_types = [i32], omp.composite}
    !$omp end teams
end subroutine distribute_parallel_do

subroutine parallel_do
!CHECK: %[[I:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFparallel_doEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[X:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFparallel_doEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel {
!CHECK: %[[LINEAR_STEP:.*]] = arith.constant 2 : i32
!CHECK: %{{.*}} = arith.constant 1 : i32
!CHECK: %{{.*}} = arith.constant 1 : i32
!DEFAULT: omp.wsloop linear(%[[X]]#0 : !fir.ref<i32> = {{.*}}) {
!OPENMP52: omp.wsloop linear(val(%[[X]]#0 : !fir.ref<i32> = {{.*}})) {
!CHECK: omp.simd linear({{.*}}) private(@_QFparallel_doEi_private_i32 {{.*}}) {
    integer :: x
    !$omp parallel do simd linear(x:2)
    do i = 1, N
    end do
    !$omp end parallel do simd
!CHECK: } {linear_var_types = [i32], omp.composite}
!CHECK: } {linear_var_types = [i32], omp.composite}
end subroutine parallel_do

subroutine teams_distribute
!CHECK: %[[I:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFteams_distributeEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[X:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFteams_distributeEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.teams {
!CHECK: %[[LINEAR_STEP:.*]] = arith.constant 1 : i32
!CHECK: {{.*}} = arith.constant 1 : i32
!CHECK: %[[IV_STEP:.*]] = arith.constant 1 : i32
!CHECK: omp.distribute {
!DEFAULT: omp.simd linear(%[[X]]#0 : !fir.ref<i32> = %[[LINEAR_STEP]] : i32, %[[I]]#0 : !fir.ref<i32> = %[[IV_STEP]] : i32) private(@_QFteams_distributeEi_private_i32 %[[I]]#0 -> %arg0 : !fir.ref<i32>) {
!OPENMP52: omp.simd linear(val(%[[X]]#0 : !fir.ref<i32> = %[[LINEAR_STEP]] : i32), val(%[[I]]#0 : !fir.ref<i32> = %[[IV_STEP]] : i32)) private(@_QFteams_distributeEi_private_i32 %[[I]]#0 -> %arg0 : !fir.ref<i32>) {
    integer :: x
    !$omp teams distribute simd linear(x)
    do i = 1, N
    end do
    !$omp end teams distribute simd
!CHECK: } {linear_var_types = [i32, i32], omp.composite}
end subroutine teams_distribute

subroutine teams_distribute_parallel_do
!CHECK: %[[I:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFteams_distribute_parallel_doEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[X:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFteams_distribute_parallel_doEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.teams {
!CHECK: omp.parallel {
!CHECK: %[[LINEAR_STEP:.*]] = arith.constant 1 : i32
!CHECK: %{{.*}} = arith.constant 1 : i32
!CHECK: %[[IV_STEP:.*]] = arith.constant 1 : i32
!CHECK: omp.distribute {
!DEFAULT: omp.wsloop linear(%[[X]]#0 : !fir.ref<i32> = {{.*}}) {
!OPENMP52: omp.wsloop linear(val(%[[X]]#0 : !fir.ref<i32> = {{.*}})) {
!CHECK: omp.simd linear({{.*}}) private(@_QFteams_distribute_parallel_doEi_private_i32 {{.*}}) {
    integer :: x
    !$omp teams distribute parallel do simd linear(x)
    do i = 1, N
    end do
    !$omp end teams distribute parallel do simd
!CHECK: } {linear_var_types = [i32], omp.composite}
!CHECK: } {linear_var_types = [i32], omp.composite}
end subroutine teams_distribute_parallel_do
