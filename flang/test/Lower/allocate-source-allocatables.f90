! RUN: %flang_fc1 -emit-fir -O2 %s -o - | FileCheck %s

! Test lowering of allocatables for allocate statements with source.

! CHECK-LABEL: func.func @_QPtest_allocatable_scalar(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "a"}) {
! CHECK-DAG:     %[[FALSE:.*]] = arith.constant false
! CHECK-DAG:     %[[ABSENT:.*]] = fir.absent !fir.box<none>
! CHECK-DAG:     %[[A_DECL:.*]] = fir.declare %[[VAL_0]] {{.*}}
! CHECK-DAG:     %[[X1:.*]] = fir.address_of(@_QFtest_allocatable_scalarEx1) : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK-DAG:     %[[X1_DECL:.*]] = fir.declare %[[X1]] {{.*}}
! CHECK-DAG:     %[[X2:.*]] = fir.address_of(@_QFtest_allocatable_scalarEx2) : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK-DAG:     %[[X2_DECL:.*]] = fir.declare %[[X2]] {{.*}}
! CHECK:         %[[EMBOX_A:.*]] = fir.embox %[[A_DECL]] : (!fir.ref<f32>) -> !fir.box<f32>
! CHECK:         %[[X1_BOX_NONE:.*]] = fir.convert %[[X1_DECL]] : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[A_BOX_NONE:.*]] = fir.convert %[[EMBOX_A]] : (!fir.box<f32>) -> !fir.box<none>
! CHECK:         %[[RES1:.*]] = fir.call @_FortranAAllocatableAllocateSource(%[[X1_BOX_NONE]], %[[A_BOX_NONE]], %[[FALSE]], %[[ABSENT]], %{{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         %[[X2_BOX_NONE:.*]] = fir.convert %[[X2_DECL]] : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[RES2:.*]] = fir.call @_FortranAAllocatableAllocateSource(%[[X2_BOX_NONE]], %[[A_BOX_NONE]], %[[FALSE]], %[[ABSENT]], %{{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         return
! CHECK:       }

subroutine test_allocatable_scalar(a)
  real, save, allocatable :: x1, x2
  real :: a

  allocate(x1, x2, source = a)
end

! CHECK-LABEL: func.func @_QPtest_allocatable_2d_array(
! CHECK-SAME:                                          %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:                                          %[[VAL_1:.*]]: !fir.ref<!fir.array<?x?xi32>> {fir.bindc_name = "a"}) {
! CHECK-DAG:     %[[FALSE:.*]] = arith.constant false
! CHECK-DAG:     %[[ABSENT:.*]] = fir.absent !fir.box<none>
! CHECK-DAG:     %[[X1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>> {bindc_name = "x1", uniq_name = "_QFtest_allocatable_2d_arrayEx1"}
! CHECK-DAG:     %[[X1_DECL:.*]] = fir.declare %[[X1]] {{.*}}
! CHECK-DAG:     %[[X2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>> {bindc_name = "x2", uniq_name = "_QFtest_allocatable_2d_arrayEx2"}
! CHECK-DAG:     %[[X2_DECL:.*]] = fir.declare %[[X2]] {{.*}}
! CHECK-DAG:     %[[X3:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>> {bindc_name = "x3", uniq_name = "_QFtest_allocatable_2d_arrayEx3"}
! CHECK-DAG:     %[[X3_DECL:.*]] = fir.declare %[[X3]] {{.*}}
! CHECK-DAG:     %[[A_DECL:.*]] = fir.declare %[[VAL_1]](%{{.*}}) {{.*}}
! CHECK:         fir.embox %[[A_DECL]]
! CHECK:         %[[X1_BOX_NONE:.*]] = fir.convert %[[X1_DECL]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         fir.call @_FortranAAllocatableSetBounds(%[[X1_BOX_NONE]], {{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:         fir.call @_FortranAAllocatableSetBounds(%[[X1_BOX_NONE]], {{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:         %[[A_BOX_NONE:.*]] = fir.convert %{{.*}} : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
! CHECK:         fir.call @_FortranAAllocatableAllocateSource(%[[X1_BOX_NONE]], %[[A_BOX_NONE]], %[[FALSE]], %[[ABSENT]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         %[[X2_BOX_NONE:.*]] = fir.convert %[[X2_DECL]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         fir.call @_FortranAAllocatableSetBounds(%[[X2_BOX_NONE]], {{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:         fir.call @_FortranAAllocatableSetBounds(%[[X2_BOX_NONE]], {{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:         fir.call @_FortranAAllocatableAllocateSource(%[[X2_BOX_NONE]], %[[A_BOX_NONE]], %[[FALSE]], %[[ABSENT]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         %[[A_SLICE_BOX_NONE:.*]] = fir.convert %{{.*}} : (!fir.box<!fir.array<{{.*}}>>) -> !fir.box<none>
! CHECK:         fir.call @_FortranAAllocatableAllocateSource(%{{.*}}, %[[A_SLICE_BOX_NONE]], %{{.*}})

subroutine test_allocatable_2d_array(n, a)
  integer, allocatable :: x1(:,:), x2(:,:), x3(:,:)
  integer :: n, sss, a(n, n)

  allocate(x1, x2, source = a)
  allocate(x3, source = a(1:3:2, 2:3), stat=sss)
end

! CHECK-LABEL: func.func @_QPtest_allocatable_with_shapespec(
! CHECK-SAME:                                                %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:                                                %[[VAL_1:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                                                %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "m"}) {
! CHECK-DAG:     %[[X1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "x1", uniq_name = "_QFtest_allocatable_with_shapespecEx1"}
! CHECK-DAG:     %[[X1_DECL:.*]] = fir.declare %[[X1]] {{.*}}
! CHECK-DAG:     %[[X2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "x2", uniq_name = "_QFtest_allocatable_with_shapespecEx2"}
! CHECK-DAG:     %[[X2_DECL:.*]] = fir.declare %[[X2]] {{.*}}
! CHECK-DAG:     %[[A_DECL:.*]] = fir.declare %[[VAL_1]](%{{.*}}) {{.*}}
! CHECK:         fir.embox %[[A_DECL]]
! CHECK:         %[[X1_BOX_NONE:.*]] = fir.convert %[[X1_DECL]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         fir.call @_FortranAAllocatableSetBounds(%[[X1_BOX_NONE]], {{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:         %[[A_BOX_NONE:.*]] = fir.convert %{{.*}} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK:         fir.call @_FortranAAllocatableAllocateSource(%[[X1_BOX_NONE]], %[[A_BOX_NONE]], %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         %[[X2_BOX_NONE:.*]] = fir.convert %[[X2_DECL]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         fir.call @_FortranAAllocatableSetBounds(%[[X2_BOX_NONE]], {{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:         fir.call @_FortranAAllocatableAllocateSource(%[[X2_BOX_NONE]], %[[A_BOX_NONE]], %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

subroutine test_allocatable_with_shapespec(n, a, m)
  integer, allocatable :: x1(:), x2(:)
  integer :: n, m, a(n)

  allocate(x1(2:m), x2(n), source = a)
end

! CHECK-LABEL: func.func @_QPtest_allocatable_from_const(
! CHECK-SAME:                                            %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:                                            %[[VAL_1:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "a"}) {
! CHECK-DAG:     %[[X1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "x1", uniq_name = "_QFtest_allocatable_from_constEx1"}
! CHECK-DAG:     %[[X1_DECL:.*]] = fir.declare %[[X1]] {{.*}}
! CHECK-DAG:     %[[CONST:.*]] = fir.address_of(@_QQro.5xi4.0) : !fir.ref<!fir.array<5xi32>>
! CHECK-DAG:     %[[CONST_DECL:.*]] = fir.declare %[[CONST]](%{{.*}}) {{.*}}
! CHECK:         %[[EMBOX_CONST:.*]] = fir.embox %[[CONST_DECL]](%{{.*}}) : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xi32>>
! CHECK:         %[[X1_BOX_NONE:.*]] = fir.convert %[[X1_DECL]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         fir.call @_FortranAAllocatableSetBounds(%[[X1_BOX_NONE]], {{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:         %[[CONST_BOX_NONE:.*]] = fir.convert %[[EMBOX_CONST]] : (!fir.box<!fir.array<5xi32>>) -> !fir.box<none>
! CHECK:         fir.call @_FortranAAllocatableAllocateSource(%[[X1_BOX_NONE]], %[[CONST_BOX_NONE]], %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         return
! CHECK:       }

subroutine test_allocatable_from_const(n, a)
  integer, allocatable :: x1(:)
  integer :: n, a(n)

  allocate(x1, source = [1, 2, 3, 4, 5])
end

! CHECK-LABEL: func.func @_QPtest_allocatable_chararray(
! CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:                                           %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "a"}) {
! CHECK-DAG:     %[[X1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>> {bindc_name = "x1", uniq_name = "_QFtest_allocatable_chararrayEx1"}
! CHECK-DAG:     %[[X1_DECL:.*]] = fir.declare %[[X1]] {{.*}}
! CHECK-DAG:     %[[UNBOX:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-DAG:     %[[A_CAST:.*]] = fir.convert %[[UNBOX]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
! CHECK-DAG:     %[[A_DECL:.*]] = fir.declare %[[A_CAST]](%{{.*}}) typeparams %[[UNBOX]]#1 {{.*}}
! CHECK:         fir.embox %[[A_DECL]]
! CHECK:         %[[X1_BOX_NONE:.*]] = fir.convert %[[X1_DECL]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         fir.call @_FortranAAllocatableSetBounds(%[[X1_BOX_NONE]], {{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:         %[[A_BOX_NONE:.*]] = fir.convert %{{.*}} : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<none>
! CHECK:         fir.call @_FortranAAllocatableAllocateSource(%[[X1_BOX_NONE]], %[[A_BOX_NONE]], %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

subroutine test_allocatable_chararray(n, a)
  character(4), allocatable :: x1(:)
  integer :: n
  character(*) :: a(n)

  allocate(x1, source = a)
end

! CHECK-LABEL: func.func @_QPtest_allocatable_char(
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:                                      %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "a"}) {
! CHECK-DAG:     %[[X1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = "x1", uniq_name = "_QFtest_allocatable_charEx1"}
! CHECK-DAG:     %[[X1_DECL:.*]] = fir.declare %[[X1]] {{.*}}
! CHECK-DAG:     %[[UNBOX:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-DAG:     %[[A_DECL:.*]] = fir.declare %[[UNBOX]]#0 typeparams %[[UNBOX]]#1 {{.*}}
! CHECK:         fir.embox %[[A_DECL]]
! CHECK:         %[[X1_BOX_NONE:.*]] = fir.convert %[[X1_DECL]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         fir.call @_FortranAAllocatableInitCharacterForAllocate(%[[X1_BOX_NONE]], %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i64, i32, i32, i32) -> ()
! CHECK:         %[[A_BOX_NONE:.*]] = fir.convert %{{.*}} : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:         fir.call @_FortranAAllocatableAllocateSource(%[[X1_BOX_NONE]], %[[A_BOX_NONE]], %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

subroutine test_allocatable_char(n, a)
  character(:), allocatable :: x1
  integer :: n
  character(*) :: a

  allocate(x1, source = a)
end

! CHECK-LABEL: func.func @_QPtest_allocatable_derived_type(
! CHECK-SAME:                                              %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>> {fir.bindc_name = "y"}) {
! CHECK-DAG:     %[[Z:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>> {bindc_name = "z", uniq_name = "_QFtest_allocatable_derived_typeEz"}
! CHECK-DAG:     %[[Z_DECL:.*]] = fir.declare %[[Z]] {{.*}}
! CHECK-DAG:     %[[Y_DECL:.*]] = fir.declare %[[VAL_0]] {{.*}}
! CHECK:         fir.load %[[Y_DECL]]
! CHECK:         %[[Z_BOX_NONE:.*]] = fir.convert %[[Z_DECL]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         fir.call @_FortranAAllocatableSetBounds(%[[Z_BOX_NONE]], {{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:         %[[Y_BOX_NONE:.*]] = fir.convert %{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>) -> !fir.box<none>
! CHECK:         fir.call @_FortranAAllocatableAllocateSource(%[[Z_BOX_NONE]], %[[Y_BOX_NONE]], %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

subroutine test_allocatable_derived_type(y)
  type t
    integer, allocatable :: x(:)
  end type
  type(t), allocatable :: z(:), y(:)

  allocate(z, source=y)
end
