! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test that IO item list are lowered and passed correctly

! CHECK-LABEL: func @_QPpass_assumed_len_char_unformatted_io(
! CHECK-SAME: %[[C_ARG:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine pass_assumed_len_char_unformatted_io(c)
  character(*) :: c
  ! CHECK: %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[UNBOX:.*]]:2 = fir.unboxchar %[[C_ARG]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[UNBOX]]#0 typeparams %[[UNBOX]]#1 dummy_scope %[[DSCOPE]] arg 1 {uniq_name = "_QFpass_assumed_len_char_unformatted_ioEc"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  write(1, rec=1) c
  ! CHECK: %[[EMBOX:.*]] = fir.embox %[[C_DECL]]#1 typeparams %[[UNBOX]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  ! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[EMBOX]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[BOX_NONE]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
end

! CHECK-LABEL: func @_QPpass_assumed_len_char_array(
! CHECK-SAME: %[[CARRAY_ARG:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine pass_assumed_len_char_array(carray)
  character(*) :: carray(2, 3)
  ! CHECK: %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[UNBOX:.*]]:2 = fir.unboxchar %[[CARRAY_ARG]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[CONV:.*]] = fir.convert %[[UNBOX]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<2x3x!fir.char<1,?>>>
  ! CHECK: %[[C2:.*]] = arith.constant 2 : index
  ! CHECK: %[[C3:.*]] = arith.constant 3 : index
  ! CHECK: %[[SHAPE:.*]] = fir.shape %[[C2]], %[[C3]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[CARRAY_DECL:.*]]:2 = hlfir.declare %[[CONV]](%[[SHAPE]]) typeparams %[[UNBOX]]#1 dummy_scope %[[DSCOPE]] arg 1 {uniq_name = "_QFpass_assumed_len_char_arrayEcarray"} : (!fir.ref<!fir.array<2x3x!fir.char<1,?>>>, !fir.shape<2>, index, !fir.dscope) -> (!fir.box<!fir.array<2x3x!fir.char<1,?>>>, !fir.ref<!fir.array<2x3x!fir.char<1,?>>>)
  print *, carray
  ! CHECK: %{{.*}} = fir.call @_FortranAioBeginExternalListOutput({{.*}})
  ! CHECK: %[[SHAPE2:.*]] = fir.shape %[[C2]], %[[C3]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[EMBOX:.*]] = fir.embox %[[CARRAY_DECL]]#1(%[[SHAPE2]]) typeparams %[[UNBOX]]#1 : (!fir.ref<!fir.array<2x3x!fir.char<1,?>>>, !fir.shape<2>, index) -> !fir.box<!fir.array<2x3x!fir.char<1,?>>>
  ! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[EMBOX]] : (!fir.box<!fir.array<2x3x!fir.char<1,?>>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[BOX_NONE]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
end

! CHECK-LABEL: func @_QPpass_array_slice_read(
! CHECK-SAME: %[[X_ARG:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
subroutine pass_array_slice_read(x)
  real :: x(:)
  ! CHECK: %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X_ARG]] dummy_scope %[[DSCOPE]] arg 1 {uniq_name = "_QFpass_array_slice_readEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  read(5, *) x(101:200:2)
  ! CHECK: %[[C101:.*]] = arith.constant 101 : index
  ! CHECK: %[[C200:.*]] = arith.constant 200 : index
  ! CHECK: %[[C2:.*]] = arith.constant 2 : index
  ! CHECK: %[[C50:.*]] = arith.constant 50 : index
  ! CHECK: %[[SHAPE:.*]] = fir.shape %[[C50]] : (index) -> !fir.shape<1>
  ! CHECK: %[[DESIGNATE:.*]] = hlfir.designate %[[X_DECL]]#0 (%[[C101]]:%[[C200]]:%[[C2]])  shape %[[SHAPE]] : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<50xf32>>
  ! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[DESIGNATE]] : (!fir.box<!fir.array<50xf32>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioInputDescriptor(%{{.*}}, %[[BOX_NONE]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
end

! CHECK-LABEL: func @_QPpass_array_slice_write(
! CHECK-SAME: %[[X_ARG:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
subroutine pass_array_slice_write(x)
  real :: x(:)
  ! CHECK: %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X_ARG]] dummy_scope %[[DSCOPE]] arg 1 {uniq_name = "_QFpass_array_slice_writeEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  write(1, rec=1) x(101:200:2)
  ! CHECK: %[[C101:.*]] = arith.constant 101 : index
  ! CHECK: %[[C200:.*]] = arith.constant 200 : index
  ! CHECK: %[[C2:.*]] = arith.constant 2 : index
  ! CHECK: %[[C50:.*]] = arith.constant 50 : index
  ! CHECK: %[[SHAPE:.*]] = fir.shape %[[C50]] : (index) -> !fir.shape<1>
  ! CHECK: %[[DESIGNATE:.*]] = hlfir.designate %[[X_DECL]]#0 (%[[C101]]:%[[C200]]:%[[C2]])  shape %[[SHAPE]] : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<50xf32>>
  ! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[DESIGNATE]] : (!fir.box<!fir.array<50xf32>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[BOX_NONE]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
end


! CHECK-LABEL: func @_QPpass_vector_subscript_write(
! CHECK-SAME: %[[X_ARG:.*]]: !fir.ref<!fir.array<100xf32>>{{.*}}, %[[J_ARG:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}})
subroutine pass_vector_subscript_write(x, j)
  ! Check that a temp is made for array with vector subscript in output IO.
  integer :: j(10)
  real :: x(100)
  ! CHECK: %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[C10:.*]] = arith.constant 10 : index
  ! CHECK: %[[J_SHAPE:.*]] = fir.shape %[[C10]] : (index) -> !fir.shape<1>
  ! CHECK: %[[J_DECL:.*]]:2 = hlfir.declare %[[J_ARG]](%[[J_SHAPE]]) dummy_scope %[[DSCOPE]] arg 2 {uniq_name = "_QFpass_vector_subscript_writeEj"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  ! CHECK: %[[C100:.*]] = arith.constant 100 : index
  ! CHECK: %[[X_SHAPE:.*]] = fir.shape %[[C100]] : (index) -> !fir.shape<1>
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X_ARG]](%[[X_SHAPE]]) dummy_scope %[[DSCOPE]] arg 1 {uniq_name = "_QFpass_vector_subscript_writeEx"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  ! CHECK: %[[VECTOR_I64:.*]] = hlfir.elemental %[[J_SHAPE]] {{.*}} -> !hlfir.expr<10xi64> {
  ! CHECK: ^bb0(%[[I:.*]]: index):
  ! CHECK:   %[[J_I_ADDR:.*]] = hlfir.designate %[[J_DECL]]#0 (%[[I]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  ! CHECK:   %[[J_I:.*]] = fir.load %[[J_I_ADDR]] : !fir.ref<i32>
  ! CHECK:   %[[J_I_I64:.*]] = fir.convert %[[J_I]] : (i32) -> i64
  ! CHECK:   hlfir.yield_element %[[J_I_I64]] : i64
  ! CHECK: }
  ! CHECK: %[[J_SHAPE2:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[VECTOR_F32:.*]] = hlfir.elemental %[[J_SHAPE2]] {{.*}} -> !hlfir.expr<10xf32> {
  ! CHECK: ^bb0(%[[I:.*]]: index):
  ! CHECK:   %[[IDX:.*]] = hlfir.apply %[[VECTOR_I64]], %[[I]] : (!hlfir.expr<10xi64>, index) -> i64
  ! CHECK:   %[[X_I_ADDR:.*]] = hlfir.designate %[[X_DECL]]#0 (%[[IDX]])  : (!fir.ref<!fir.array<100xf32>>, i64) -> !fir.ref<f32>
  ! CHECK:   %[[X_I:.*]] = fir.load %[[X_I_ADDR]] : !fir.ref<f32>
  ! CHECK:   hlfir.yield_element %[[X_I]] : f32
  ! CHECK: }
  ! CHECK: %[[ASSOC:.*]]:3 = hlfir.associate %[[VECTOR_F32]](%[[J_SHAPE2]]) {{.*}} : (!hlfir.expr<10xf32>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>, i1)
  ! CHECK: %[[J_SHAPE3:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[EMBOX:.*]] = fir.embox %[[ASSOC]]#0(%[[J_SHAPE3]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
  ! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[EMBOX]] : (!fir.box<!fir.array<10xf32>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[BOX_NONE]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
  ! CHECK: hlfir.end_associate %[[ASSOC]]#1, %[[ASSOC]]#2 : !fir.ref<!fir.array<10xf32>>, i1
  ! CHECK: hlfir.destroy %[[VECTOR_F32]] : !hlfir.expr<10xf32>
  ! CHECK: hlfir.destroy %[[VECTOR_I64]] : !hlfir.expr<10xi64>
  write(1, rec=1) x(j)
end
