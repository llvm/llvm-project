! Test COPYPRIVATE.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 \
! RUN: | FileCheck %s

!CHECK-DAG: func private @_copy_i64(%{{.*}}: !fir.ref<i64>, %{{.*}}: !fir.ref<i64>)
!CHECK-DAG: func private @_copy_f32(%{{.*}}: !fir.ref<f32>, %{{.*}}: !fir.ref<f32>)
!CHECK-DAG: func private @_copy_f64(%{{.*}}: !fir.ref<f64>, %{{.*}}: !fir.ref<f64>)
!CHECK-DAG: func private @_copy_z32(%{{.*}}: !fir.ref<!fir.complex<4>>, %{{.*}}: !fir.ref<!fir.complex<4>>)
!CHECK-DAG: func private @_copy_z64(%{{.*}}: !fir.ref<!fir.complex<8>>, %{{.*}}: !fir.ref<!fir.complex<8>>)
!CHECK-DAG: func private @_copy_l32(%{{.*}}: !fir.ref<!fir.logical<4>>, %{{.*}}: !fir.ref<!fir.logical<4>>)
!CHECK-DAG: func private @_copy_l64(%{{.*}}: !fir.ref<!fir.logical<8>>, %{{.*}}: !fir.ref<!fir.logical<8>>)
!CHECK-DAG: func private @_copy_c8x3(%{{.*}}: !fir.ref<!fir.char<1,3>>, %{{.*}}: !fir.ref<!fir.char<1,3>>)
!CHECK-DAG: func private @_copy_c8x8(%{{.*}}: !fir.ref<!fir.char<1,8>>, %{{.*}}: !fir.ref<!fir.char<1,8>>)
!CHECK-DAG: func private @_copy_c16x8(%{{.*}}: !fir.ref<!fir.char<2,8>>, %{{.*}}: !fir.ref<!fir.char<2,8>>)

!CHECK-DAG: func private @_copy_box_Uxi32(%{{.*}}: !fir.ref<!fir.box<!fir.array<?xi32>>>, %{{.*}}: !fir.ref<!fir.box<!fir.array<?xi32>>>)
!CHECK-DAG: func private @_copy_10xi32(%{{.*}}: !fir.ref<!fir.array<10xi32>>, %{{.*}}: !fir.ref<!fir.array<10xi32>>)
!CHECK-DAG: func private @_copy_3x4xi32(%{{.*}}: !fir.ref<!fir.array<3x4xi32>>, %{{.*}}: !fir.ref<!fir.array<3x4xi32>>)
!CHECK-DAG: func private @_copy_10xf32(%{{.*}}: !fir.ref<!fir.array<10xf32>>, %{{.*}}: !fir.ref<!fir.array<10xf32>>)
!CHECK-DAG: func private @_copy_3x4xz32(%{{.*}}: !fir.ref<!fir.array<3x4x!fir.complex<4>>>, %{{.*}}: !fir.ref<!fir.array<3x4x!fir.complex<4>>>)
!CHECK-DAG: func private @_copy_10xl32(%{{.*}}: !fir.ref<!fir.array<10x!fir.logical<4>>>, %{{.*}}: !fir.ref<!fir.array<10x!fir.logical<4>>>)
!CHECK-DAG: func private @_copy_3xc8x8(%{{.*}}: !fir.ref<!fir.array<3x!fir.char<1,8>>>, %{{.*}}: !fir.ref<!fir.array<3x!fir.char<1,8>>>)
!CHECK-DAG: func private @_copy_3xc16x5(%{{.*}}: !fir.ref<!fir.array<3x!fir.char<2,5>>>, %{{.*}}: !fir.ref<!fir.array<3x!fir.char<2,5>>>)

!CHECK-DAG: func private @_copy_rec__QFtest_dtTdt(%{{.*}}: !fir.ref<!fir.type<_QFtest_dtTdt{i:i32,r:f32}>>, %{{.*}}: !fir.ref<!fir.type<_QFtest_dtTdt{i:i32,r:f32}>>)
!CHECK-DAG: func private @_copy_box_heap_Uxi32(%{{.*}}: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %{{.*}}: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
!CHECK-DAG: func private @_copy_box_heap_i32(%{{.*}}: !fir.ref<!fir.box<!fir.heap<i32>>>, %{{.*}}: !fir.ref<!fir.box<!fir.heap<i32>>>)
!CHECK-DAG: func private @_copy_box_ptr_i32(%{{.*}}: !fir.ref<!fir.box<!fir.ptr<i32>>>, %{{.*}}: !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK-DAG: func private @_copy_box_ptr_Uxf32(%{{.*}}: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, %{{.*}}: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
!CHECK-DAG: func private @_copy_box_heap_Uxc8x5(%{{.*}}: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,5>>>>>, %{{.*}}: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,5>>>>>)
!CHECK-DAG: func private @_copy_box_ptr_Uxc8x9(%{{.*}}: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,9>>>>>, %{{.*}}: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,9>>>>>)

!CHECK-LABEL: func private @_copy_i32(
!CHECK-SAME:                  %[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<i32>) {
!CHECK-NEXT:    %[[DST:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_copy_i32_dst"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:    %[[SRC:.*]]:2 = hlfir.declare %[[ARG1]] {uniq_name = "_copy_i32_src"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:    %[[SRC_VAL:.*]] = fir.load %[[SRC]]#0 : !fir.ref<i32>
!CHECK-NEXT:    hlfir.assign %[[SRC_VAL]] to %[[DST]]#0 : i32, !fir.ref<i32>
!CHECK-NEXT:    return
!CHECK-NEXT:  }

!CHECK-LABEL: func @_QPtest_tp
!CHECK:         omp.parallel
!CHECK:           %[[I:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_tpEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[J:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_tpEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[K:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_tpEk"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:           omp.single copyprivate(%[[I]]#0 -> @_copy_i32 : !fir.ref<i32>, %[[J]]#0 -> @_copy_i32 : !fir.ref<i32>, %[[K]]#0 -> @_copy_f32 : !fir.ref<f32>)
subroutine test_tp()
  integer, save :: i, j
  !$omp threadprivate(i, j)
  real :: k

  k = 33.3
  !$omp parallel firstprivate(k)
  !$omp single
  i = 11
  j = 22
  !$omp end single copyprivate(i, j, k)
  !$omp end parallel
end subroutine

!CHECK-LABEL: func @_QPtest_scalar
!CHECK:         omp.parallel
!CHECK:           %[[I1:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_scalarEi1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[I2:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_scalarEi2"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
!CHECK:           %[[I3:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_scalarEi3"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
!CHECK:           %[[R1:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_scalarEr1"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:           %[[R2:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_scalarEr2"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
!CHECK:           %[[C1:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_scalarEc1"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
!CHECK:           %[[C2:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_scalarEc2"} : (!fir.ref<!fir.complex<8>>) -> (!fir.ref<!fir.complex<8>>, !fir.ref<!fir.complex<8>>)
!CHECK:           %[[L1:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_scalarEl1"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK:           %[[L2:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_scalarEl2"} : (!fir.ref<!fir.logical<8>>) -> (!fir.ref<!fir.logical<8>>, !fir.ref<!fir.logical<8>>)
!CHECK:           %[[S1:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_scalarEs1"} : (!fir.ref<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>)
!CHECK:           %[[S2:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_scalarEs2"} : (!fir.ref<!fir.char<1,8>>, index) -> (!fir.ref<!fir.char<1,8>>, !fir.ref<!fir.char<1,8>>)
!CHECK:           %[[S3:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_scalarEs3"} : (!fir.ref<!fir.char<2,8>>, index) -> (!fir.ref<!fir.char<2,8>>, !fir.ref<!fir.char<2,8>>)
!CHECK:           omp.single copyprivate(%[[I1]]#0 -> @_copy_i32 : !fir.ref<i32>, %[[I2]]#0 -> @_copy_i64 : !fir.ref<i64>, %[[I3]]#0 -> @_copy_i64 : !fir.ref<i64>, %[[R1]]#0 -> @_copy_f32 : !fir.ref<f32>, %[[R2]]#0 -> @_copy_f64 : !fir.ref<f64>, %[[C1]]#0 -> @_copy_z32 : !fir.ref<!fir.complex<4>>, %[[C2]]#0 -> @_copy_z64 : !fir.ref<!fir.complex<8>>, %[[L1]]#0 -> @_copy_l32 : !fir.ref<!fir.logical<4>>, %[[L2]]#0 -> @_copy_l64 : !fir.ref<!fir.logical<8>>, %[[S1]]#0 -> @_copy_c8x3 : !fir.ref<!fir.char<1,3>>, %[[S2]]#0 -> @_copy_c8x8 : !fir.ref<!fir.char<1,8>>, %[[S3]]#0 -> @_copy_c16x8 : !fir.ref<!fir.char<2,8>>)
subroutine test_scalar()
  integer(4) :: i1
  integer(8) :: i2, i3
  real(4) :: r1
  real(8) :: r2
  complex(4) :: c1
  complex(8) :: c2
  logical(4) :: l1
  logical(8) :: l2
  character(kind=1, len=3) :: s1
  character(kind=1, len=8) :: s2
  character(kind=2, len=8) :: s3

  !$omp parallel private(i1, i2, i3, r1, r2, c1, c2, l1, l2, s1, s2, s3)
  !$omp single
  !$omp end single copyprivate(i1, i2, i3, r1, r2, c1, c2, l1, l2, s1, s2, s3)
  !$omp end parallel
end subroutine

!CHECK-LABEL: func @_QPtest_array
!CHECK:         omp.parallel
!CHECK:           %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtest_arrayEa"} : (!fir.box<!fir.array<?xi32>>, !fir.shift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
!CHECK:           %[[I1:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtest_arrayEi1"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
!CHECK:           %[[I2:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtest_arrayEi2"} : (!fir.ref<!fir.array<3x4xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x4xi32>>, !fir.ref<!fir.array<3x4xi32>>)
!CHECK:           %[[I3:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtest_arrayEi3"} : (!fir.ref<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>)
!CHECK:           %[[R1:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtest_arrayEr1"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
!CHECK:           %[[C1:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtest_arrayEc1"} : (!fir.ref<!fir.array<3x4x!fir.complex<4>>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x4x!fir.complex<4>>>, !fir.ref<!fir.array<3x4x!fir.complex<4>>>)
!CHECK:           %[[L1:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtest_arrayEl1"} : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.ref<!fir.array<10x!fir.logical<4>>>)
!CHECK:           %[[S1:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFtest_arrayEs1"} : (!fir.ref<!fir.array<3x!fir.char<1,8>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<3x!fir.char<1,8>>>, !fir.ref<!fir.array<3x!fir.char<1,8>>>)
!CHECK:           %[[S2:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFtest_arrayEs2"} : (!fir.ref<!fir.array<3x!fir.char<2,5>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<3x!fir.char<2,5>>>, !fir.ref<!fir.array<3x!fir.char<2,5>>>)
!CHECK:           %[[A_REF:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
!CHECK:           fir.store %[[A]]#0 to %[[A_REF]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
!CHECK:           %[[I3_REF:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
!CHECK:           fir.store %[[I3]]#0 to %[[I3_REF]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
!CHECK:           omp.single copyprivate(%[[A_REF]] -> @_copy_box_Uxi32 : !fir.ref<!fir.box<!fir.array<?xi32>>>, %[[I1]]#0 -> @_copy_10xi32 : !fir.ref<!fir.array<10xi32>>, %[[I2]]#0 -> @_copy_3x4xi32 : !fir.ref<!fir.array<3x4xi32>>, %[[I3_REF]] -> @_copy_box_Uxi32 : !fir.ref<!fir.box<!fir.array<?xi32>>>, %[[R1]]#0 -> @_copy_10xf32 : !fir.ref<!fir.array<10xf32>>, %[[C1]]#0 -> @_copy_3x4xz32 : !fir.ref<!fir.array<3x4x!fir.complex<4>>>, %[[L1]]#0 -> @_copy_10xl32 : !fir.ref<!fir.array<10x!fir.logical<4>>>, %[[S1]]#0 -> @_copy_3xc8x8 : !fir.ref<!fir.array<3x!fir.char<1,8>>>, %[[S2]]#0 -> @_copy_3xc16x5 : !fir.ref<!fir.array<3x!fir.char<2,5>>>)
subroutine test_array(a, n)
  integer :: a(:), n
  integer :: i1(10), i2(3, 4), i3(n)
  real :: r1(10)
  complex :: c1(3, 4)
  logical :: l1(10)
  character(8) :: s1(3)
  character(kind=2, len=5) :: s2(3)

  !$omp parallel private(a, i1, i2, i3, r1, c1, l1, s1, s2)
  !$omp single
  !$omp end single copyprivate(a, i1, i2, i3, r1, c1, l1, s1, s2)
  !$omp end parallel
end subroutine

!CHECK-LABEL: func @_QPtest_dt
!CHECK:         omp.parallel
!CHECK:           %[[T:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_dtEt"} : (!fir.ref<!fir.type<_QFtest_dtTdt{i:i32,r:f32}>>) -> (!fir.ref<!fir.type<_QFtest_dtTdt{i:i32,r:f32}>>, !fir.ref<!fir.type<_QFtest_dtTdt{i:i32,r:f32}>>)
!CHECK:           omp.single copyprivate(%[[T]]#0 -> @_copy_rec__QFtest_dtTdt : !fir.ref<!fir.type<_QFtest_dtTdt{i:i32,r:f32}>>)
subroutine test_dt()
  type dt
    integer :: i
    real :: r
  end type
  type(dt) :: t

  !$omp parallel private(t)
  !$omp single
  !$omp end single copyprivate(t)
  !$omp end parallel
end subroutine

!CHECK-LABEL: func @_QPtest_attr
!CHECK:         omp.parallel
!CHECK:           %[[I1:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_attrEi1"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
!CHECK:           %[[I2:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_attrEi2"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
!CHECK:           %[[I3:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_attrEi3"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK:           %[[R1:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_attrEr1"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
!CHECK:           %[[C1:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_attrEc1"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,5>>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,5>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,5>>>>>)
!CHECK:           %[[C2:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_attrEc2"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,9>>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,9>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,9>>>>>)
!CHECK:           omp.single copyprivate(%[[I1]]#0 -> @_copy_box_heap_Uxi32 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %[[I2:.*]]#0 -> @_copy_box_heap_i32 : !fir.ref<!fir.box<!fir.heap<i32>>>, %[[I3]]#0 -> @_copy_box_ptr_i32 : !fir.ref<!fir.box<!fir.ptr<i32>>>, %[[R1]]#0 -> @_copy_box_ptr_Uxf32 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>,  %[[C1]]#0 -> @_copy_box_heap_Uxc8x5 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,5>>>>>, %[[C2]]#0 -> @_copy_box_ptr_Uxc8x9 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,9>>>>>)
subroutine test_attr()
  integer, allocatable :: i1(:)
  integer, allocatable :: i2
  integer, pointer :: i3
  real, pointer :: r1(:)
  character(kind=1, len=5), allocatable :: c1(:)
  character(kind=1, len=9), pointer :: c2(:)

  !$omp parallel private(i1, i2, i3, r1, c1, c2)
  !$omp single
  !$omp end single copyprivate(i1, i2, i3, r1, c1, c2)
  !$omp end parallel
end subroutine
