! RUN: bbc %s -o - | FileCheck %s

! Test lowering of local character variables

! CHECK-LABEL: func @_QPscalar_cst_len
subroutine scalar_cst_len()
  character(10) :: c
  ! CHECK: fir.alloca !fir.char<1,10> {{{.*}}uniq_name = "_QFscalar_cst_lenEc"}
end subroutine

! CHECK-LABEL: func @_QPscalar_dyn_len
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32>
subroutine scalar_dyn_len(l)
  integer :: l
  character(l) :: c
  ! CHECK: %[[lexpr:.*]] = fir.load %[[arg0]] : !fir.ref<i32>
  ! CHECK: %[[is_positive:.*]] = arith.cmpi sgt, %[[lexpr]], %c0{{.*}} : i32
  ! CHECK: %[[l:.*]] = arith.select %[[is_positive]], %[[lexpr]], %c0{{.*}} : i32
  ! CHECK: fir.alloca !fir.char<1,?>(%[[l]] : i32) {{{.*}}uniq_name = "_QFscalar_dyn_lenEc"}
end subroutine

! CHECK-LABEL: func @_QPcst_array_cst_len
subroutine cst_array_cst_len()
  character(10) :: c(20)
  ! CHECK: fir.alloca !fir.array<20x!fir.char<1,10>> {{{.*}}uniq_name = "_QFcst_array_cst_lenEc"}
end subroutine

! CHECK-LABEL: func @_QPcst_array_dyn_len
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32>
subroutine cst_array_dyn_len(l)
  integer :: l
  character(l) :: c(10)
  ! CHECK: %[[lexpr:.*]] = fir.load %[[arg0]] : !fir.ref<i32>
  ! CHECK: %[[is_positive:.*]] = arith.cmpi sgt, %[[lexpr]], %c0{{.*}} : i32
  ! CHECK: %[[l:.*]] = arith.select %[[is_positive]], %[[lexpr]], %c0{{.*}} : i32
  ! CHECK: fir.alloca !fir.array<10x!fir.char<1,?>>(%[[l]] : i32) {{{.*}}uniq_name = "_QFcst_array_dyn_lenEc"}
end subroutine

! CHECK-LABEL: func @_QPdyn_array_cst_len
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32>
subroutine dyn_array_cst_len(n)
  integer :: n
  character(10) :: c(n)
  ! CHECK: %[[n:.*]] = fir.load %[[arg0]] : !fir.ref<i32>
  ! CHECK: %[[ni:.*]] = fir.convert %[[n]] : (i32) -> index
  ! CHECK: %[[is_positive:.*]] = arith.cmpi sgt, %[[ni]], %c0{{.*}} : index
  ! CHECK: %[[extent:.*]] = arith.select %[[is_positive]], %[[ni]], %c0{{.*}} : index
  ! CHECK: fir.alloca !fir.array<?x!fir.char<1,10>>, %[[extent]] {{{.*}}uniq_name = "_QFdyn_array_cst_lenEc"}
end subroutine

! CHECK: func @_QPdyn_array_dyn_len
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>
subroutine dyn_array_dyn_len(l, n)
  integer :: l, n
  character(l) :: c(n)
  ! CHECK-DAG: %[[lexpr:.*]] = fir.load %[[arg0]] : !fir.ref<i32>
  ! CHECK-DAG: %[[is_positive:.*]] = arith.cmpi sgt, %[[lexpr]], %c0{{.*}} : i32
  ! CHECK-DAG: %[[l:.*]] = arith.select %[[is_positive]], %[[lexpr]], %c0{{.*}} : i32
  ! CHECK-DAG: %[[n:.*]] = fir.load %[[arg1]] : !fir.ref<i32>
  ! CHECK: %[[ni:.*]] = fir.convert %[[n]] : (i32) -> index
  ! CHECK: %[[is_positive:.*]] = arith.cmpi sgt, %[[ni]], %c0{{.*}} : index
  ! CHECK: %[[extent:.*]] = arith.select %[[is_positive]], %[[ni]], %c0{{.*}} : index
  ! CHECK: fir.alloca !fir.array<?x!fir.char<1,?>>(%[[l]] : i32), %[[extent]] {{{.*}}uniq_name = "_QFdyn_array_dyn_lenEc"}
end subroutine

! CHECK-LABEL: func @_QPcst_array_cst_len_lb
subroutine cst_array_cst_len_lb()
  character(10) :: c(11:30)
  ! CHECK: fir.alloca !fir.array<20x!fir.char<1,10>> {{{.*}}uniq_name = "_QFcst_array_cst_len_lbEc"}
end subroutine

! CHECK-LABEL: func @_QPcst_array_dyn_len_lb
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i64>
subroutine cst_array_dyn_len_lb(l)
  integer(8) :: l
  character(l) :: c(11:20)
  ! CHECK: %[[lexpr:.*]] = fir.load %[[arg0]] : !fir.ref<i64>
  ! CHECK: %[[is_positive:.*]] = arith.cmpi sgt, %[[lexpr]], %c0{{.*}} : i64
  ! CHECK: %[[l:.*]] = arith.select %[[is_positive]], %[[lexpr]], %c0{{.*}} : i64
  ! CHECK: fir.alloca !fir.array<10x!fir.char<1,?>>(%[[l]] : i64) {{{.*}}uniq_name = "_QFcst_array_dyn_len_lbEc"}
end subroutine

! CHECK-LABEL: func @_QPdyn_array_cst_len_lb
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i64>
subroutine dyn_array_cst_len_lb(n)
  integer(8) :: n
  character(10) :: c(11:n)
  ! CHECK-DAG: %[[cm10:.*]] = arith.constant -10 : index
  ! CHECK-DAG: %[[n:.*]] = fir.load %[[arg0]] : !fir.ref<i64>
  ! CHECK-DAG: %[[ni:.*]] = fir.convert %[[n]] : (i64) -> index
  ! CHECK: %[[raw_extent:.*]] = arith.addi %[[ni]], %[[cm10]] : index
  ! CHECK: %[[is_positive:.*]] = arith.cmpi sgt, %[[raw_extent]], %c0{{.*}} : index
  ! CHECK: %[[extent:.*]] = arith.select %[[is_positive]], %[[raw_extent]], %c0{{.*}} : index
  ! CHECK: fir.alloca !fir.array<?x!fir.char<1,10>>, %[[extent]] {{{.*}}uniq_name = "_QFdyn_array_cst_len_lbEc"}
end subroutine

! CHECK-LABEL: func @_QPdyn_array_dyn_len_lb
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i64>{{.*}}, %[[arg1:.*]]: !fir.ref<i64>
subroutine dyn_array_dyn_len_lb(l, n)
  integer(8) :: l, n
  character(l) :: c(11:n)
  ! CHECK-DAG: %[[cm10:.*]] = arith.constant -10 : index
  ! CHECK-DAG: %[[lexpr:.*]] = fir.load %[[arg0]] : !fir.ref<i64>
  ! CHECK-DAG: %[[is_positive:.*]] = arith.cmpi sgt, %[[lexpr]], %c0{{.*}} : i64
  ! CHECK-DAG: %[[l:.*]] = arith.select %[[is_positive]], %[[lexpr]], %c0{{.*}} : i64
  ! CHECK-DAG: %[[n:.*]] = fir.load %[[arg1]] : !fir.ref<i64>
  ! CHECK-DAG: %[[ni:.*]] = fir.convert %[[n]] : (i64) -> index
  ! CHECK: %[[raw_extent:.*]] = arith.addi %[[ni]], %[[cm10]] : index
  ! CHECK: %[[is_positive:.*]] = arith.cmpi sgt, %[[raw_extent]], %c0{{.*}} : index
  ! CHECK: %[[extent:.*]] = arith.select %[[is_positive]], %[[raw_extent]], %c0{{.*}} : index
  ! CHECK: fir.alloca !fir.array<?x!fir.char<1,?>>(%[[l]] : i64), %[[extent]] {{{.*}}uniq_name = "_QFdyn_array_dyn_len_lbEc"}
end subroutine

! Test that the length of assumed length parameter is correctly deduced in lowering.
! CHECK-LABEL: func @_QPassumed_length_param
subroutine assumed_length_param(n)
  character(*), parameter :: c(1)=(/"abcd"/)
  integer :: n
  ! CHECK: %[[c4:.*]] = arith.constant 4 : i64
  ! CHECK: fir.store %[[c4]] to %[[tmp:.*]] : !fir.ref<i64>
  ! CHECK: fir.call @_QPtake_int(%[[tmp]]) {{.*}}: (!fir.ref<i64>) -> ()
  call take_int(len(c(n), kind=8))
end

! CHECK-LABEL: func @_QPscalar_cst_neg_len
subroutine scalar_cst_neg_len()
  character(-1) :: c
  ! CHECK: fir.alloca !fir.char<1,0> {{{.*}}uniq_name = "_QFscalar_cst_neg_lenEc"}
end subroutine
