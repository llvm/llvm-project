! RUN: bbc %s -o - | FileCheck %s

! Test lowering of local character variables

! CHECK-LABEL: func @_QPscalar_cst_len
subroutine scalar_cst_len()
  character(10) :: c
  ! CHECK: fir.alloca !fir.char<1,10> {{{.*}}uniq_name = "_QFscalar_cst_lenEc"}
end subroutine

! CHECK-LABEL: func @_QPscalar_dyn_len
subroutine scalar_dyn_len(l)
  integer :: l
  character(l) :: c
  ! CHECK: %[[l:.*]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK: fir.alloca !fir.char<1,?>(%[[l]] : i32) {{{.*}}uniq_name = "_QFscalar_dyn_lenEc"}
end subroutine

! CHECK-LABEL: func @_QPcst_array_cst_len
subroutine cst_array_cst_len()
  character(10) :: c(20)
  ! CHECK: fir.alloca !fir.array<20x!fir.char<1,10>> {{{.*}}uniq_name = "_QFcst_array_cst_lenEc"}
end subroutine

! CHECK-LABEL: func @_QPcst_array_dyn_len
subroutine cst_array_dyn_len(l)
  integer :: l
  character(l) :: c
  ! CHECK: %[[l:.*]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK: fir.alloca !fir.char<1,?>(%[[l]] : i32) {{{.*}}uniq_name = "_QFcst_array_dyn_lenEc"}
end subroutine

! CHECK-LABEL: func @_QPdyn_array_cst_len
subroutine dyn_array_cst_len(n)
  integer :: n
  character(10) :: c(n)
  ! CHECK: %[[n:.*]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK: %[[ni:.*]] = fir.convert %[[n]] : (i32) -> index
  ! CHECK: fir.alloca !fir.array<?x!fir.char<1,10>>, %[[ni]] {{{.*}}uniq_name = "_QFdyn_array_cst_lenEc"}
end subroutine

! CHECK: func @_QPdyn_array_dyn_len
subroutine dyn_array_dyn_len(l, n)
  integer :: l, n
  character(l) :: c(n)
  ! CHECK-DAG: %[[l:.*]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK-DAG: %[[n:.*]] = fir.load %arg1 : !fir.ref<i32>
  ! CHECK: %[[ni:.*]] = fir.convert %[[n]] : (i32) -> index
  ! CHECK: fir.alloca !fir.array<?x!fir.char<1,?>>(%[[l]] : i32), %[[ni]] {{{.*}}uniq_name = "_QFdyn_array_dyn_lenEc"}
end subroutine
