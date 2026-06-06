! Test lowering of IO input items with vector subscripts
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s
! UNSUPPORTED: system-windows

! CHECK-LABEL: func @_QPsimple(
! CHECK-SAME: %[[VAL_X_ARG:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_Y_ARG:.*]]: !fir.ref<!fir.array<3xi32>>{{.*}}) {
subroutine simple(x, y)
  integer :: y(3)
  integer :: x(10)
  read(*,*) x(y)
! CHECK-DAG: %[[VAL_C10:.*]] = arith.constant 10 : index
! CHECK-DAG: %[[VAL_C3:.*]] = arith.constant 3 : index
! CHECK-DAG: %[[VAL_X:.*]]:2 = hlfir.declare %[[VAL_X_ARG]](%{{.*}}){{.*}}{uniq_name = "_QFsimpleEx"}
! CHECK-DAG: %[[VAL_Y:.*]]:2 = hlfir.declare %[[VAL_Y_ARG]](%{{.*}}){{.*}}{uniq_name = "_QFsimpleEy"}
! CHECK-DAG: %[[VAL_5:.*]] = arith.constant 5 : i32
! CHECK:   %[[VAL_7:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:   %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_9:.*]] = fir.call @_FortranAioBeginExternalListInput(%[[VAL_5]], %[[VAL_8]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[VAL_C10]] : (index) -> !fir.shape<1>
! CHECK:   %[[VAL_11:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index) -> !fir.slice<1>
! CHECK:   %[[VAL_BOUND:.*]] = arith.subi %{{.*}}, %{{.*}} : index
! CHECK:   fir.do_loop %[[IDX:.*]] = %{{.*}} to %[[VAL_BOUND]] step %{{.*}} {
! CHECK:     %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_Y]]#0, %[[IDX]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[VAL_17:.*]] = fir.load %[[VAL_15]] : !fir.ref<i32>
! CHECK:     %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
! CHECK:     %[[VAL_19:.*]] = fir.array_coor %[[VAL_X]]#0(%[[SHAPE]]) {{\[}}%[[VAL_11]]] %[[VAL_18]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<i32>
! CHECK:     %[[VAL_21:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<i32>) -> !fir.ref<i64>
! CHECK:     %[[VAL_22:.*]] = fir.call @_FortranAioInputInteger(%[[VAL_9]], %[[VAL_21]], %{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i64>, i32) -> i1
! CHECK:   }
! CHECK:   %[[VAL_25:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_9]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPonly_once(
! CHECK-SAME: %[[VAL_X_ARG:.*]]: !fir.box<!fir.array<?x?xf32>>{{.*}}) {
subroutine only_once(x)
  interface
    function get_vector()
      integer, allocatable :: get_vector(:)
    end function
    integer function get_substcript()
    end function
  end interface
  real :: x(:, :)
  ! Test subscripts are only evaluated once.
  read(*,*) x(get_substcript(), get_vector())
! CHECK:   %[[VAL_RES:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = ".result"}
! CHECK:   %[[VAL_X:.*]]:2 = hlfir.declare %[[VAL_X_ARG]]{{.*}}{uniq_name = "_QFonly_onceEx"}
! CHECK:   %[[VAL_BEGIN:.*]] = fir.call @_FortranAioBeginExternalListInput({{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_SUB:.*]] = fir.call @_QPget_substcript() {{.*}}: () -> i32
! CHECK:   %[[VAL_SUB_I64:.*]] = fir.convert %[[VAL_SUB]] : (i32) -> i64
! CHECK:   %[[VAL_RES_DECL:.*]]:2 = hlfir.declare %[[VAL_RES]] {uniq_name = ".tmp.func_result"}
! CHECK:   %[[VAL_GETVEC:.*]] = fir.call @_QPget_vector() {{.*}}: () -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:   fir.save_result %[[VAL_GETVEC]] to %[[VAL_RES_DECL]]#0
! CHECK:   %[[VAL_LOAD:.*]] = fir.load %[[VAL_RES_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:   %[[VAL_EXPR:.*]] = hlfir.as_expr %[[VAL_LOAD]] move %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, i1) -> !hlfir.expr<?xi32>
! CHECK:   %[[VAL_DIMS:.*]]:3 = fir.box_dims %[[VAL_LOAD]], %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:   %[[VAL_SHAPE:.*]] = fir.shape %[[VAL_DIMS]]#1 : (index) -> !fir.shape<1>
! CHECK:   %[[VAL_ASSOC:.*]]:3 = hlfir.associate %[[VAL_EXPR]](%[[VAL_SHAPE]]) {adapt.valuebyref} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
! CHECK:   %[[VAL_SLICE:.*]] = fir.slice %[[VAL_SUB_I64]], %{{.*}}, %{{.*}}, %{{.*}}, %[[VAL_DIMS]]#1, %{{.*}} : (i64, index, index, index, index, index) -> !fir.slice<2>
! CHECK:   fir.do_loop %[[IDX:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
! CHECK:     %[[SUB_IDX:.*]] = fir.convert %[[VAL_SUB_I64]] : (i64) -> index
! CHECK:     %[[VEL:.*]] = fir.coordinate_of %[[VAL_ASSOC]]#1, %[[IDX]] : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[VEL_LD:.*]] = fir.load %[[VEL]] : !fir.ref<i32>
! CHECK:     %[[VEL_IDX:.*]] = fir.convert %[[VEL_LD]] : (i32) -> index
! CHECK:     %[[ELEM:.*]] = fir.array_coor %[[VAL_X]]#1 {{\[}}%[[VAL_SLICE]]] %[[SUB_IDX]], %[[VEL_IDX]] : (!fir.box<!fir.array<?x?xf32>>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:     %{{.*}} = fir.call @_FortranAioInputReal32(%[[VAL_BEGIN]], %[[ELEM]]) {{.*}}: (!fir.ref<i8>, !fir.ref<f32>) -> i1
! CHECK:   }
! CHECK:   hlfir.end_associate %[[VAL_ASSOC]]#1, %[[VAL_ASSOC]]#2 : !fir.ref<!fir.array<?xi32>>, i1
! CHECK:   hlfir.destroy %[[VAL_EXPR]] : !hlfir.expr<?xi32>
! CHECK:   %{{.*}} = fir.call @_FortranAioEndIoStatement(%[[VAL_BEGIN]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPwith_assumed_shapes(
! CHECK-SAME: %[[VAL_X_ARG:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[VAL_Y_ARG:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) {
subroutine with_assumed_shapes(x, y)
  integer :: y(:)
  integer :: x(:)
  read(*,*) x(y)
! CHECK:   %[[VAL_X:.*]]:2 = hlfir.declare %[[VAL_X_ARG]]{{.*}}{uniq_name = "_QFwith_assumed_shapesEx"}
! CHECK:   %[[VAL_Y:.*]]:2 = hlfir.declare %[[VAL_Y_ARG]]{{.*}}{uniq_name = "_QFwith_assumed_shapesEy"}
! CHECK:   %[[VAL_BEGIN:.*]] = fir.call @_FortranAioBeginExternalListInput({{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_DIMS:.*]]:3 = fir.box_dims %[[VAL_Y]]#1, %{{.*}} : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:   %[[VAL_C1:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_SLICE:.*]] = fir.slice %[[VAL_C1]], %[[VAL_DIMS]]#1, %[[VAL_C1]] : (index, index, index) -> !fir.slice<1>
! CHECK:   fir.do_loop %[[IDX:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
! CHECK:     %[[VEL:.*]] = fir.coordinate_of %[[VAL_Y]]#1, %[[IDX]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[VEL_LD:.*]] = fir.load %[[VEL]] : !fir.ref<i32>
! CHECK:     %[[VEL_IDX:.*]] = fir.convert %[[VEL_LD]] : (i32) -> index
! CHECK:     %[[ELEM:.*]] = fir.array_coor %[[VAL_X]]#1 {{\[}}%[[VAL_SLICE]]] %[[VEL_IDX]] : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
! CHECK:     %[[VAL_79:.*]] = fir.convert %[[ELEM]] : (!fir.ref<i32>) -> !fir.ref<i64>
! CHECK:     %{{.*}} = fir.call @_FortranAioInputInteger(%[[VAL_BEGIN]], %[[VAL_79]], %{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i64>, i32) -> i1
! CHECK:   }
! CHECK:   %{{.*}} = fir.call @_FortranAioEndIoStatement(%[[VAL_BEGIN]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPlower_bounds(
! CHECK-SAME: %[[VAL_X_ARG:.*]]: !fir.ref<!fir.array<4x6xi32>>{{.*}}, %[[VAL_Y_ARG:.*]]: !fir.ref<!fir.array<3xi32>>{{.*}}) {
subroutine lower_bounds(x, y)
  integer :: y(3)
  integer :: x(2:5,3:8)
  read(*,*) x(3, y)
! CHECK:   %[[VAL_SS:.*]] = fir.shape_shift %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:   %[[VAL_X:.*]]:2 = hlfir.declare %[[VAL_X_ARG]](%[[VAL_SS]]){{.*}}{uniq_name = "_QFlower_boundsEx"}
! CHECK:   %[[VAL_Y:.*]]:2 = hlfir.declare %[[VAL_Y_ARG]](%{{.*}}){{.*}}{uniq_name = "_QFlower_boundsEy"}
! CHECK:   %[[VAL_BEGIN:.*]] = fir.call @_FortranAioBeginExternalListInput({{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_C3I64:.*]] = arith.constant 3 : i64
! CHECK:   %[[VAL_SS2:.*]] = fir.shape_shift %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:   %[[VAL_UNDEF:.*]] = fir.undefined index
! CHECK:   %[[VAL_SLICE:.*]] = fir.slice %[[VAL_C3I64]], %[[VAL_UNDEF]], %[[VAL_UNDEF]], %{{.*}}, %{{.*}}, %{{.*}} : (i64, index, index, index, index, index) -> !fir.slice<2>
! CHECK:   fir.do_loop %[[IDX:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
! CHECK:     %[[VAL_C3IDX:.*]] = fir.convert %[[VAL_C3I64]] : (i64) -> index
! CHECK:     %[[VEL:.*]] = fir.coordinate_of %[[VAL_Y]]#0, %[[IDX]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[VEL_LD:.*]] = fir.load %[[VEL]] : !fir.ref<i32>
! CHECK:     %[[VEL_IDX:.*]] = fir.convert %[[VEL_LD]] : (i32) -> index
! CHECK:     %[[ELEM:.*]] = fir.array_coor %[[VAL_X]]#1(%[[VAL_SS2]]) {{\[}}%[[VAL_SLICE]]] %[[VAL_C3IDX]], %[[VEL_IDX]] : (!fir.ref<!fir.array<4x6xi32>>, !fir.shapeshift<2>, !fir.slice<2>, index, index) -> !fir.ref<i32>
! CHECK:     %{{.*}} = fir.convert %[[ELEM]] : (!fir.ref<i32>) -> !fir.ref<i64>
! CHECK:     %{{.*}} = fir.call @_FortranAioInputInteger({{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i64>, i32) -> i1
! CHECK:   }
! CHECK:   %{{.*}} = fir.call @_FortranAioEndIoStatement(%[[VAL_BEGIN]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPtwo_vectors(
! CHECK-SAME: %[[VAL_X_ARG:.*]]: !fir.ref<!fir.array<4x4xf32>>{{.*}}, %[[VAL_Y1_ARG:.*]]: !fir.ref<!fir.array<3xi32>>{{.*}}, %[[VAL_Y2_ARG:.*]]: !fir.ref<!fir.array<3xi32>>{{.*}}) {
subroutine two_vectors(x, y1, y2)
  integer :: y1(3), y2(3)
  real :: x(4, 4)
  read(*,*) x(y1, y2)
! CHECK:   %[[VAL_X:.*]]:2 = hlfir.declare %[[VAL_X_ARG]](%{{.*}}){{.*}}{uniq_name = "_QFtwo_vectorsEx"}
! CHECK:   %[[VAL_Y1:.*]]:2 = hlfir.declare %[[VAL_Y1_ARG]](%{{.*}}){{.*}}{uniq_name = "_QFtwo_vectorsEy1"}
! CHECK:   %[[VAL_Y2:.*]]:2 = hlfir.declare %[[VAL_Y2_ARG]](%{{.*}}){{.*}}{uniq_name = "_QFtwo_vectorsEy2"}
! CHECK:   %[[VAL_BEGIN:.*]] = fir.call @_FortranAioBeginExternalListInput({{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_SHAPE:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
! CHECK:   %[[VAL_SLICE:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:   fir.do_loop %[[IDX_OUTER:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
! CHECK:     fir.do_loop %[[IDX_INNER:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
! CHECK:       %[[Y1_EL:.*]] = fir.coordinate_of %[[VAL_Y1]]#0, %[[IDX_INNER]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:       %[[Y1_LD:.*]] = fir.load %[[Y1_EL]] : !fir.ref<i32>
! CHECK:       %[[Y1_IDX:.*]] = fir.convert %[[Y1_LD]] : (i32) -> index
! CHECK:       %[[Y2_EL:.*]] = fir.coordinate_of %[[VAL_Y2]]#0, %[[IDX_OUTER]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:       %[[Y2_LD:.*]] = fir.load %[[Y2_EL]] : !fir.ref<i32>
! CHECK:       %[[Y2_IDX:.*]] = fir.convert %[[Y2_LD]] : (i32) -> index
! CHECK:       %[[ELEM:.*]] = fir.array_coor %[[VAL_X]]#0(%[[VAL_SHAPE]]) {{\[}}%[[VAL_SLICE]]] %[[Y1_IDX]], %[[Y2_IDX]] : (!fir.ref<!fir.array<4x4xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:       %{{.*}} = fir.call @_FortranAioInputReal32(%[[VAL_BEGIN]], %[[ELEM]]) {{.*}}: (!fir.ref<i8>, !fir.ref<f32>) -> i1
! CHECK:     }
! CHECK:   }
! CHECK:   %{{.*}} = fir.call @_FortranAioEndIoStatement(%[[VAL_BEGIN]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPtriplets_and_vector(
! CHECK-SAME:    %[[VAL_X_ARG:.*]]: !fir.ref<!fir.array<4x4xcomplex<f32>>>{{.*}}, %[[VAL_Y_ARG:.*]]: !fir.ref<!fir.array<3xi32>>{{.*}}) {
subroutine triplets_and_vector(x, y)
  integer :: y(3)
  complex :: x(4, 4)
  read(*,*) x(1:4:2, y)
! CHECK:   %[[VAL_X:.*]]:2 = hlfir.declare %[[VAL_X_ARG]](%{{.*}}){{.*}}{uniq_name = "_QFtriplets_and_vectorEx"}
! CHECK:   %[[VAL_Y:.*]]:2 = hlfir.declare %[[VAL_Y_ARG]](%{{.*}}){{.*}}{uniq_name = "_QFtriplets_and_vectorEy"}
! CHECK:   %[[VAL_BEGIN:.*]] = fir.call @_FortranAioBeginExternalListInput({{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_SHAPE:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
! CHECK:   %[[VAL_SLICE:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:   fir.do_loop %[[IDX_OUTER:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
! CHECK:     fir.do_loop %[[IDX_INNER:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
! CHECK:       %[[Y_EL:.*]] = fir.coordinate_of %[[VAL_Y]]#0, %[[IDX_OUTER]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:       %[[Y_LD:.*]] = fir.load %[[Y_EL]] : !fir.ref<i32>
! CHECK:       %[[Y_IDX:.*]] = fir.convert %[[Y_LD]] : (i32) -> index
! CHECK:       %[[ELEM:.*]] = fir.array_coor %[[VAL_X]]#0(%[[VAL_SHAPE]]) {{\[}}%[[VAL_SLICE]]] %[[IDX_INNER]], %[[Y_IDX]] : (!fir.ref<!fir.array<4x4xcomplex<f32>>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<complex<f32>>
! CHECK:       %[[ELEM_F32:.*]] = fir.convert %[[ELEM]] : (!fir.ref<complex<f32>>) -> !fir.ref<f32>
! CHECK:       %{{.*}} = fir.call @_FortranAioInputComplex32(%[[VAL_BEGIN]], %[[ELEM_F32]]) {{.*}}: (!fir.ref<i8>, !fir.ref<f32>) -> i1
! CHECK:     }
! CHECK:   }
! CHECK:   %{{.*}} = fir.call @_FortranAioEndIoStatement(%[[VAL_BEGIN]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPsimple_char(
! CHECK-SAME: %[[VAL_X_ARG:.*]]: !fir.boxchar<1>{{.*}}, %[[VAL_Y_ARG:.*]]: !fir.ref<!fir.array<3xi32>>{{.*}}) {
subroutine simple_char(x, y)
  integer :: y(3)
  character(*) :: x(3:8)
  read(*,*) x(y)
! CHECK:   %[[VAL_UNBOX:.*]]:2 = fir.unboxchar %[[VAL_X_ARG]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:   %[[VAL_X_CAST:.*]] = fir.convert %[[VAL_UNBOX]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<6x!fir.char<1,?>>>
! CHECK:   %[[VAL_X:.*]]:2 = hlfir.declare %[[VAL_X_CAST]](%{{.*}}) typeparams %[[VAL_UNBOX]]#1{{.*}}{uniq_name = "_QFsimple_charEx"}
! CHECK:   %[[VAL_Y:.*]]:2 = hlfir.declare %[[VAL_Y_ARG]](%{{.*}}){{.*}}{uniq_name = "_QFsimple_charEy"}
! CHECK:   %[[VAL_BEGIN:.*]] = fir.call @_FortranAioBeginExternalListInput({{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_SS:.*]] = fir.shape_shift %{{.*}}, %{{.*}} : (index, index) -> !fir.shapeshift<1>
! CHECK:   %[[VAL_SLICE:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index) -> !fir.slice<1>
! CHECK:   fir.do_loop %[[IDX:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
! CHECK:     %[[VEL:.*]] = fir.coordinate_of %[[VAL_Y]]#0, %[[IDX]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[VEL_LD:.*]] = fir.load %[[VEL]] : !fir.ref<i32>
! CHECK:     %[[VEL_IDX:.*]] = fir.convert %[[VEL_LD]] : (i32) -> index
! CHECK:     %[[ELEM:.*]] = fir.array_coor %[[VAL_X]]#1(%[[VAL_SS]]) {{\[}}%[[VAL_SLICE]]] %[[VEL_IDX]] typeparams %[[VAL_UNBOX]]#1 : (!fir.ref<!fir.array<6x!fir.char<1,?>>>, !fir.shapeshift<1>, !fir.slice<1>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:     %[[ELEM_I8:.*]] = fir.convert %[[ELEM]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:     %[[LEN_I64:.*]] = fir.convert %[[VAL_UNBOX]]#1 : (index) -> i64
! CHECK:     %{{.*}} = fir.call @_FortranAioInputAscii(%[[VAL_BEGIN]], %[[ELEM_I8]], %[[LEN_I64]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:   }
! CHECK:   %{{.*}} = fir.call @_FortranAioEndIoStatement(%[[VAL_BEGIN]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPsubstring(
! CHECK-SAME: %[[VAL_X_ARG:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>{{.*}}, %[[VAL_Y_ARG:.*]]: !fir.ref<!fir.array<3xi32>>{{.*}}, %[[VAL_I_ARG:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_J_ARG:.*]]: !fir.ref<i32>{{.*}}) {
subroutine substring(x, y, i, j)
  integer :: y(3), i, j
  character(*) :: x(:)
  read(*,*) x(y)(i:j)
! CHECK:   %[[VAL_I:.*]]:2 = hlfir.declare %[[VAL_I_ARG]]{{.*}}{uniq_name = "_QFsubstringEi"}
! CHECK:   %[[VAL_J:.*]]:2 = hlfir.declare %[[VAL_J_ARG]]{{.*}}{uniq_name = "_QFsubstringEj"}
! CHECK:   %[[VAL_X:.*]]:2 = hlfir.declare %[[VAL_X_ARG]]{{.*}}{uniq_name = "_QFsubstringEx"}
! CHECK:   %[[VAL_Y:.*]]:2 = hlfir.declare %[[VAL_Y_ARG]](%{{.*}}){{.*}}{uniq_name = "_QFsubstringEy"}
! CHECK:   %[[VAL_BEGIN:.*]] = fir.call @_FortranAioBeginExternalListInput({{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_I_LD:.*]] = fir.load %[[VAL_I]]#0 : !fir.ref<i32>
! CHECK:   %[[VAL_I_I64:.*]] = fir.convert %[[VAL_I_LD]] : (i32) -> i64
! CHECK:   %[[VAL_I_IDX:.*]] = fir.convert %[[VAL_I_I64]] : (i64) -> index
! CHECK:   %[[VAL_J_LD:.*]] = fir.load %[[VAL_J]]#0 : !fir.ref<i32>
! CHECK:   %[[VAL_J_I64:.*]] = fir.convert %[[VAL_J_LD]] : (i32) -> i64
! CHECK:   %[[VAL_J_IDX:.*]] = fir.convert %[[VAL_J_I64]] : (i64) -> index
! CHECK:   %[[VAL_SLICE:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index) -> !fir.slice<1>
! CHECK:   fir.do_loop %[[IDX:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
! CHECK:     %[[VEL:.*]] = fir.coordinate_of %[[VAL_Y]]#0, %[[IDX]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[VEL_LD:.*]] = fir.load %[[VEL]] : !fir.ref<i32>
! CHECK:     %[[VEL_IDX:.*]] = fir.convert %[[VEL_LD]] : (i32) -> index
! CHECK:     %[[ELEM:.*]] = fir.array_coor %[[VAL_X]]#1 {{\[}}%[[VAL_SLICE]]] %[[VEL_IDX]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.slice<1>, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:     %[[OFFSET:.*]] = arith.subi %[[VAL_I_IDX]], %{{.*}} : index
! CHECK:     %[[ELEM_AS_ARR:.*]] = fir.convert %[[ELEM]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:     %[[SUBSTR:.*]] = fir.coordinate_of %[[ELEM_AS_ARR]], %[[OFFSET]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:     %[[SUBSTR_AS_C1Q:.*]] = fir.convert %[[SUBSTR]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:     %[[LEN_DIFF:.*]] = arith.subi %[[VAL_J_IDX]], %[[VAL_I_IDX]] : index
! CHECK:     %[[LEN_PLUS_1:.*]] = arith.addi %[[LEN_DIFF]], %{{.*}} : index
! CHECK:     %[[LT0:.*]] = arith.cmpi slt, %[[LEN_PLUS_1]], %{{.*}} : index
! CHECK:     %[[LEN:.*]] = arith.select %[[LT0]], %{{.*}}, %[[LEN_PLUS_1]] : index
! CHECK:     %[[ELEM_I8:.*]] = fir.convert %[[SUBSTR_AS_C1Q]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:     %[[LEN_I64:.*]] = fir.convert %[[LEN]] : (index) -> i64
! CHECK:     %{{.*}} = fir.call @_FortranAioInputAscii(%[[VAL_BEGIN]], %[[ELEM_I8]], %[[LEN_I64]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:   }
! CHECK:   %{{.*}} = fir.call @_FortranAioEndIoStatement(%[[VAL_BEGIN]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPcomplex_part(
! CHECK-SAME: %[[VAL_Z_ARG:.*]]: !fir.box<!fir.array<?xcomplex<f32>>>{{.*}}, %[[VAL_Y_ARG:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) {
subroutine complex_part(z, y)
  integer :: y(:)
  complex :: z(:)
  read(*,*) z(y)%IM
! CHECK:   %[[VAL_Y:.*]]:2 = hlfir.declare %[[VAL_Y_ARG]]{{.*}}{uniq_name = "_QFcomplex_partEy"}
! CHECK:   %[[VAL_Z:.*]]:2 = hlfir.declare %[[VAL_Z_ARG]]{{.*}}{uniq_name = "_QFcomplex_partEz"}
! CHECK:   %[[VAL_BEGIN:.*]] = fir.call @_FortranAioBeginExternalListInput({{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_DIMS:.*]]:3 = fir.box_dims %[[VAL_Y]]#1, %{{.*}} : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:   %[[VAL_C1I32:.*]] = arith.constant 1 : i32
! CHECK:   %[[VAL_SLICE:.*]] = fir.slice %{{.*}}, %[[VAL_DIMS]]#1, %{{.*}} path %[[VAL_C1I32]] : (index, index, index, i32) -> !fir.slice<1>
! CHECK:   fir.do_loop %[[IDX:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
! CHECK:     %[[VEL:.*]] = fir.coordinate_of %[[VAL_Y]]#1, %[[IDX]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[VEL_LD:.*]] = fir.load %[[VEL]] : !fir.ref<i32>
! CHECK:     %[[VEL_IDX:.*]] = fir.convert %[[VEL_LD]] : (i32) -> index
! CHECK:     %[[ELEM:.*]] = fir.array_coor %[[VAL_Z]]#1 {{\[}}%[[VAL_SLICE]]] %[[VEL_IDX]] : (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.slice<1>, index) -> !fir.ref<f32>
! CHECK:     %{{.*}} = fir.call @_FortranAioInputReal32(%[[VAL_BEGIN]], %[[ELEM]]) {{.*}}: (!fir.ref<i8>, !fir.ref<f32>) -> i1
! CHECK:   }
! CHECK:   %{{.*}} = fir.call @_FortranAioEndIoStatement(%[[VAL_BEGIN]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

module derived_types
  type t
    integer :: i
    character(2) :: c
  end type
  type t2
    type(t) :: a(5,5)
  end type
end module

! CHECK-LABEL: func @_QPsimple_derived(
! CHECK-SAME: %[[VAL_X_ARG:.*]]: !fir.ref<!fir.array<6x!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>>{{.*}}, %[[VAL_Y_ARG:.*]]: !fir.ref<!fir.array<4xi32>>{{.*}}) {
subroutine simple_derived(x, y)
  use derived_types
  integer :: y(4)
  type(t) :: x(3:8)
  read(*,*) x(y)
! CHECK:   %[[VAL_SS:.*]] = fir.shape_shift %{{.*}}, %{{.*}} : (index, index) -> !fir.shapeshift<1>
! CHECK:   %[[VAL_X:.*]]:2 = hlfir.declare %[[VAL_X_ARG]](%[[VAL_SS]]){{.*}}{uniq_name = "_QFsimple_derivedEx"}
! CHECK:   %[[VAL_Y:.*]]:2 = hlfir.declare %[[VAL_Y_ARG]](%{{.*}}){{.*}}{uniq_name = "_QFsimple_derivedEy"}
! CHECK:   %[[VAL_BEGIN:.*]] = fir.call @_FortranAioBeginExternalListInput({{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_SS2:.*]] = fir.shape_shift %{{.*}}, %{{.*}} : (index, index) -> !fir.shapeshift<1>
! CHECK:   %[[VAL_SLICE:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index) -> !fir.slice<1>
! CHECK:   fir.do_loop %[[IDX:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
! CHECK:     %[[VEL:.*]] = fir.coordinate_of %[[VAL_Y]]#0, %[[IDX]] : (!fir.ref<!fir.array<4xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[VEL_LD:.*]] = fir.load %[[VEL]] : !fir.ref<i32>
! CHECK:     %[[VEL_IDX:.*]] = fir.convert %[[VEL_LD]] : (i32) -> index
! CHECK:     %[[ELEM:.*]] = fir.array_coor %[[VAL_X]]#1(%[[VAL_SS2]]) {{\[}}%[[VAL_SLICE]]] %[[VEL_IDX]] : (!fir.ref<!fir.array<6x!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>>, !fir.shapeshift<1>, !fir.slice<1>, index) -> !fir.ref<!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>
! CHECK:     %[[ELEM_BOX:.*]] = fir.embox %[[ELEM]] : (!fir.ref<!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>) -> !fir.box<!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>
! CHECK:     %[[ELEM_BOX_NONE:.*]] = fir.convert %[[ELEM_BOX]] : (!fir.box<!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>) -> !fir.box<none>
! CHECK:     %{{.*}} = fir.call @_FortranAioInputDerivedType(%[[VAL_BEGIN]], %[[ELEM_BOX_NONE]], {{.*}}) {{.*}}: (!fir.ref<i8>, !fir.box<none>, !fir.ref<none>) -> i1
! CHECK:   }
! CHECK:   %{{.*}} = fir.call @_FortranAioEndIoStatement(%[[VAL_BEGIN]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPwith_path(
! CHECK-SAME: %[[VAL_B_ARG:.*]]: !fir.box<!fir.array<?x?x?x!fir.type<_QMderived_typesTt2{a:!fir.array<5x5x!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>}>>>{{.*}}, %[[VAL_I_ARG:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) {
subroutine with_path(b, i)
  use derived_types
  type(t2) :: b(4:, 4:, 4:)
  integer :: i(:)
  read (*, *) b(5, i, 8:9:1)%a(4,5)%i
! CHECK:   %[[VAL_SHIFT:.*]] = fir.shift %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index) -> !fir.shift<3>
! CHECK:   %[[VAL_B:.*]]:2 = hlfir.declare %[[VAL_B_ARG]](%[[VAL_SHIFT]]){{.*}}{uniq_name = "_QFwith_pathEb"}
! CHECK:   %[[VAL_I:.*]]:2 = hlfir.declare %[[VAL_I_ARG]]{{.*}}{uniq_name = "_QFwith_pathEi"}
! CHECK:   %[[VAL_BEGIN:.*]] = fir.call @_FortranAioBeginExternalListInput({{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[VAL_DIMS:.*]]:3 = fir.box_dims %[[VAL_I]]#1, %{{.*}} : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:   %[[VAL_FIELD_A:.*]] = fir.field_index a, !fir.type<_QMderived_typesTt2{a:!fir.array<5x5x!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>}>
! CHECK:   %[[VAL_FIELD_I:.*]] = fir.field_index i, !fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>
! CHECK:   %[[VAL_SHIFT2:.*]] = fir.shift %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index) -> !fir.shift<3>
! CHECK:   %[[VAL_UNDEF:.*]] = fir.undefined index
! CHECK:   %[[VAL_SLICE:.*]] = fir.slice %{{.*}}, %[[VAL_UNDEF]], %[[VAL_UNDEF]], %{{.*}}, %[[VAL_DIMS]]#1, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} path %[[VAL_FIELD_A]], %{{.*}}, %{{.*}}, %[[VAL_FIELD_I]] : (i64, index, index, index, index, index, index, index, index, !fir.field, i64, i64, !fir.field) -> !fir.slice<3>
! CHECK:   fir.do_loop %[[IDX_OUTER:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
! CHECK:     fir.do_loop %[[IDX_INNER:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
! CHECK:       %[[FIVE_IDX:.*]] = fir.convert %{{.*}} : (i64) -> index
! CHECK:       %[[VEL:.*]] = fir.coordinate_of %[[VAL_I]]#1, %[[IDX_INNER]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:       %[[VEL_LD:.*]] = fir.load %[[VEL]] : !fir.ref<i32>
! CHECK:       %[[VEL_IDX:.*]] = fir.convert %[[VEL_LD]] : (i32) -> index
! CHECK:       %[[ELEM:.*]] = fir.array_coor %[[VAL_B]]#1(%[[VAL_SHIFT2]]) {{\[}}%[[VAL_SLICE]]] %[[FIVE_IDX]], %[[VEL_IDX]], %[[IDX_OUTER]] : (!fir.box<!fir.array<?x?x?x!fir.type<_QMderived_typesTt2{a:!fir.array<5x5x!fir.type<_QMderived_typesTt{i:i32,c:!fir.char<1,2>}>>}>>>, !fir.shift<3>, !fir.slice<3>, index, index, index) -> !fir.ref<i32>
! CHECK:       %[[ELEM_I64:.*]] = fir.convert %[[ELEM]] : (!fir.ref<i32>) -> !fir.ref<i64>
! CHECK:       %{{.*}} = fir.call @_FortranAioInputInteger(%[[VAL_BEGIN]], %[[ELEM_I64]], %{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i64>, i32) -> i1
! CHECK:     }
! CHECK:   }
! CHECK:   %{{.*}} = fir.call @_FortranAioEndIoStatement(%[[VAL_BEGIN]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPsimple_iostat(
! CHECK-SAME: %[[VAL_X_ARG:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}, %[[VAL_Y_ARG:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[VAL_J_ARG:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_STAT_ARG:.*]]: !fir.ref<i32>{{.*}}) {
subroutine simple_iostat(x, y, j, stat)
  integer :: j, y(:), stat
  real :: x(:)
  read(*, *, iostat=stat) x(y), j
! CHECK:   %[[VAL_J:.*]]:2 = hlfir.declare %[[VAL_J_ARG]]{{.*}}{uniq_name = "_QFsimple_iostatEj"}
! CHECK:   %[[VAL_STAT:.*]]:2 = hlfir.declare %[[VAL_STAT_ARG]]{{.*}}{uniq_name = "_QFsimple_iostatEstat"}
! CHECK:   %[[VAL_X:.*]]:2 = hlfir.declare %[[VAL_X_ARG]]{{.*}}{uniq_name = "_QFsimple_iostatEx"}
! CHECK:   %[[VAL_Y:.*]]:2 = hlfir.declare %[[VAL_Y_ARG]]{{.*}}{uniq_name = "_QFsimple_iostatEy"}
! CHECK:   %[[VAL_BEGIN:.*]] = fir.call @_FortranAioBeginExternalListInput({{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   fir.call @_FortranAioEnableHandlers(%[[VAL_BEGIN]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i1, i1, i1, i1, i1) -> ()
! CHECK:   %[[VAL_DIMS:.*]]:3 = fir.box_dims %[[VAL_Y]]#1, %{{.*}} : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:   %[[VAL_TRUE:.*]] = arith.constant true
! CHECK:   %[[VAL_SLICE:.*]] = fir.slice %{{.*}}, %[[VAL_DIMS]]#1, %{{.*}} : (index, index, index) -> !fir.slice<1>
! CHECK:   %[[BOUND:.*]] = arith.subi %[[VAL_DIMS]]#1, %{{.*}} : index
! CHECK:   %[[ITER_RES:.*]] = fir.iterate_while (%[[IDX:.*]] = %{{.*}} to %[[BOUND]] step %{{.*}}) and (%[[COND:.*]] = %[[VAL_TRUE]]) {
! CHECK:     %[[VEL:.*]] = fir.coordinate_of %[[VAL_Y]]#1, %[[IDX]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[VEL_LD:.*]] = fir.load %[[VEL]] : !fir.ref<i32>
! CHECK:     %[[VEL_IDX:.*]] = fir.convert %[[VEL_LD]] : (i32) -> index
! CHECK:     %[[ELEM:.*]] = fir.array_coor %[[VAL_X]]#1 {{\[}}%[[VAL_SLICE]]] %[[VEL_IDX]] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>, index) -> !fir.ref<f32>
! CHECK:     %[[OK:.*]] = fir.call @_FortranAioInputReal32(%[[VAL_BEGIN]], %[[ELEM]]) {{.*}}: (!fir.ref<i8>, !fir.ref<f32>) -> i1
! CHECK:     fir.result %[[OK]] : i1
! CHECK:   }
! CHECK:   fir.if %[[ITER_RES]] {
! CHECK:     %[[J_I64:.*]] = fir.convert %[[VAL_J]]#0 : (!fir.ref<i32>) -> !fir.ref<i64>
! CHECK:     %{{.*}} = fir.call @_FortranAioInputInteger(%[[VAL_BEGIN]], %[[J_I64]], %{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i64>, i32) -> i1
! CHECK:   }
! CHECK:   %[[END_ST:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_BEGIN]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   fir.store %[[END_ST]] to %[[VAL_STAT]]#0 : !fir.ref<i32>
! CHECK:   return
end subroutine

! CHECK-LABEL: func @_QPiostat_in_io_loop(
! CHECK-SAME: %[[VAL_K_ARG:.*]]: !fir.ref<!fir.array<3x5xi32>>{{.*}}, %[[VAL_J_ARG:.*]]: !fir.ref<!fir.array<3xi32>>{{.*}}, %[[VAL_STAT_ARG:.*]]: !fir.ref<i32>{{.*}}) {
subroutine iostat_in_io_loop(k, j, stat)
  integer :: k(3, 5)
  integer :: j(3)
  integer  :: stat
  read(*, *, iostat=stat) (k(i, j), i=1,3,1)
! CHECK:   %[[VAL_I_ALLOC:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFiostat_in_io_loopEi"}
! CHECK:   %[[VAL_I:.*]]:2 = hlfir.declare %[[VAL_I_ALLOC]] {uniq_name = "_QFiostat_in_io_loopEi"}
! CHECK:   %[[VAL_J:.*]]:2 = hlfir.declare %[[VAL_J_ARG]](%{{.*}}){{.*}}{uniq_name = "_QFiostat_in_io_loopEj"}
! CHECK:   %[[VAL_K:.*]]:2 = hlfir.declare %[[VAL_K_ARG]](%{{.*}}){{.*}}{uniq_name = "_QFiostat_in_io_loopEk"}
! CHECK:   %[[VAL_STAT:.*]]:2 = hlfir.declare %[[VAL_STAT_ARG]]{{.*}}{uniq_name = "_QFiostat_in_io_loopEstat"}
! CHECK:   %[[VAL_BEGIN:.*]] = fir.call @_FortranAioBeginExternalListInput({{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   fir.call @_FortranAioEnableHandlers(%[[VAL_BEGIN]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i1, i1, i1, i1, i1) -> ()
! CHECK:   %[[VAL_TRUE:.*]] = arith.constant true
! CHECK:   %[[OUTER:.*]]:2 = fir.iterate_while (%[[I_IDX:.*]] = %{{.*}} to %{{.*}} step %{{.*}}) and (%[[I_COND:.*]] = %[[VAL_TRUE]]) -> (index, i1) {
! CHECK:     %[[I_AS_I32:.*]] = fir.convert %[[I_IDX]] : (index) -> i32
! CHECK:     fir.store %[[I_AS_I32]] to %[[VAL_I]]#0 : !fir.ref<i32>
! CHECK:     %[[INNER_RES:.*]] = fir.if %[[I_COND]] -> (i1) {
! CHECK:       %[[I_LD:.*]] = fir.load %[[VAL_I]]#0 : !fir.ref<i32>
! CHECK:       %[[I_I64:.*]] = fir.convert %[[I_LD]] : (i32) -> i64
! CHECK:       %[[K_SHAPE:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
! CHECK:       %[[VAL_UNDEF:.*]] = fir.undefined index
! CHECK:       %[[VAL_SLICE:.*]] = fir.slice %[[I_I64]], %[[VAL_UNDEF]], %[[VAL_UNDEF]], %{{.*}}, %{{.*}}, %{{.*}} : (i64, index, index, index, index, index) -> !fir.slice<2>
! CHECK:       %[[INNER_ITER:.*]] = fir.iterate_while (%[[J_IDX:.*]] = %{{.*}} to %{{.*}} step %{{.*}}) and (%{{.*}} = %[[I_COND]]) {
! CHECK:         %[[I_AS_IDX:.*]] = fir.convert %[[I_I64]] : (i64) -> index
! CHECK:         %[[J_EL:.*]] = fir.coordinate_of %[[VAL_J]]#0, %[[J_IDX]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[J_LD:.*]] = fir.load %[[J_EL]] : !fir.ref<i32>
! CHECK:         %[[J_AS_IDX:.*]] = fir.convert %[[J_LD]] : (i32) -> index
! CHECK:         %[[ELEM:.*]] = fir.array_coor %[[VAL_K]]#0(%[[K_SHAPE]]) {{\[}}%[[VAL_SLICE]]] %[[I_AS_IDX]], %[[J_AS_IDX]] : (!fir.ref<!fir.array<3x5xi32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<i32>
! CHECK:         %[[ELEM_I64:.*]] = fir.convert %[[ELEM]] : (!fir.ref<i32>) -> !fir.ref<i64>
! CHECK:         %[[OK:.*]] = fir.call @_FortranAioInputInteger(%[[VAL_BEGIN]], %[[ELEM_I64]], %{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i64>, i32) -> i1
! CHECK:         fir.result %[[OK]] : i1
! CHECK:       }
! CHECK:       fir.result %[[INNER_ITER]] : i1
! CHECK:     } else {
! CHECK:       fir.result %{{.*}} : i1
! CHECK:     }
! CHECK:     %{{.*}} = arith.addi %[[I_IDX]], %{{.*}} overflow<nsw> : index
! CHECK:     %{{.*}} = arith.select %[[INNER_RES]], %{{.*}}, %[[I_IDX]] : index
! CHECK:     fir.result %{{.*}}, %[[INNER_RES]] : index, i1
! CHECK:   }
! CHECK:   %[[FINAL_I:.*]] = fir.convert %[[OUTER]]#0 : (index) -> i32
! CHECK:   fir.store %[[FINAL_I]] to %[[VAL_I]]#0 : !fir.ref<i32>
! CHECK:   %[[END_ST:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_BEGIN]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   fir.store %[[END_ST]] to %[[VAL_STAT]]#0 : !fir.ref<i32>
! CHECK:   return
end subroutine
