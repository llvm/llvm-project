! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPverify_test(
! CHECK-SAME: %[[S1_ARG:.*]]: !fir.boxchar<1>{{.*}}, %[[S2_ARG:.*]]: !fir.boxchar<1>{{.*}}) -> i32 {
integer function verify_test(s1, s2)
! CHECK: %[[BOX:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
! CHECK: %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[S1_UNBOX:.*]]:2 = fir.unboxchar %[[S1_ARG]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[S1_DECL:.*]]:2 = hlfir.declare %[[S1_UNBOX]]#0 typeparams %[[S1_UNBOX]]#1 dummy_scope %[[DSCOPE]] arg 1 {uniq_name = "_QFverify_testEs1"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK: %[[S2_UNBOX:.*]]:2 = fir.unboxchar %[[S2_ARG]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[S2_DECL:.*]]:2 = hlfir.declare %[[S2_UNBOX]]#0 typeparams %[[S2_UNBOX]]#1 dummy_scope %[[DSCOPE]] arg 2 {uniq_name = "_QFverify_testEs2"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK: %[[RET_VAR:.*]] = fir.alloca i32 {bindc_name = "verify_test", uniq_name = "_QFverify_testEverify_test"}
! CHECK: %[[RET_DECL:.*]]:2 = hlfir.declare %[[RET_VAR]] {uniq_name = "_QFverify_testEverify_test"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[KIND:.*]] = arith.constant 4 : i32
! CHECK: %[[ABSENT:.*]] = fir.absent !fir.box<i1>
! CHECK: %[[S1_EMBOX:.*]] = fir.embox %[[S1_DECL]]#1 typeparams %[[S1_UNBOX]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK: %[[S2_EMBOX:.*]] = fir.embox %[[S2_DECL]]#1 typeparams %[[S2_UNBOX]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK: %[[NULL:.*]] = fir.zero_bits !fir.heap<i32>
! CHECK: %[[RES_BOX:.*]] = fir.embox %[[NULL]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK: fir.store %[[RES_BOX]] to %[[BOX]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK: %[[FILE:.*]] = fir.address_of({{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK: %[[LINE:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[BOX]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[S1_NONE:.*]] = fir.convert %[[S1_EMBOX]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK: %[[S2_NONE:.*]] = fir.convert %[[S2_EMBOX]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK: %[[ABSENT_NONE:.*]] = fir.convert %[[ABSENT]] : (!fir.box<i1>) -> !fir.box<none>
! CHECK: %[[FILE_PTR:.*]] = fir.convert %[[FILE]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK: fir.call @_FortranAVerify(%[[BOX_NONE]], %[[S1_NONE]], %[[S2_NONE]], %[[ABSENT_NONE]], %[[KIND]], %[[FILE_PTR]], %[[LINE]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> ()
! CHECK: %[[LOAD_BOX:.*]] = fir.load %[[BOX]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD_BOX]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK: %[[VAL:.*]] = fir.load %[[ADDR]] : !fir.heap<i32>
! CHECK: fir.freemem %[[ADDR]] : !fir.heap<i32>
! CHECK: hlfir.assign %[[VAL]] to %[[RET_DECL]]#0 : i32, !fir.ref<i32>
! CHECK: %[[RES:.*]] = fir.load %[[RET_DECL]]#0 : !fir.ref<i32>
! CHECK: return %[[RES]] : i32
  character(*) :: s1, s2
  verify_test = verify(s1, s2, kind=4)
end function verify_test

! CHECK-LABEL: func @_QPverify_test2(
! CHECK-SAME: %[[S1_ARG:.*]]: !fir.boxchar<1>{{.*}}, %[[S2_ARG:.*]]: !fir.boxchar<1>{{.*}}) -> i32 {
integer function verify_test2(s1, s2)
! CHECK: %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[S1_UNBOX:.*]]:2 = fir.unboxchar %[[S1_ARG]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[S1_DECL:.*]]:2 = hlfir.declare %[[S1_UNBOX]]#0 typeparams %[[S1_UNBOX]]#1 dummy_scope %[[DSCOPE]] arg 1 {uniq_name = "_QFverify_test2Es1"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK: %[[S2_UNBOX:.*]]:2 = fir.unboxchar %[[S2_ARG]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[S2_DECL:.*]]:2 = hlfir.declare %[[S2_UNBOX]]#0 typeparams %[[S2_UNBOX]]#1 dummy_scope %[[DSCOPE]] arg 2 {uniq_name = "_QFverify_test2Es2"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK: %[[RET_VAR:.*]] = fir.alloca i32 {bindc_name = "verify_test2", uniq_name = "_QFverify_test2Everify_test2"}
! CHECK: %[[RET_DECL:.*]]:2 = hlfir.declare %[[RET_VAR]] {uniq_name = "_QFverify_test2Everify_test2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[BACK:.*]] = arith.constant true
! CHECK: %[[S1_PTR:.*]] = fir.convert %[[S1_DECL]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK: %[[S1_LEN:.*]] = fir.convert %[[S1_UNBOX]]#1 : (index) -> i64
! CHECK: %[[S2_PTR:.*]] = fir.convert %[[S2_DECL]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK: %[[S2_LEN:.*]] = fir.convert %[[S2_UNBOX]]#1 : (index) -> i64
! CHECK: %[[RES_I64:.*]] = fir.call @_FortranAVerify1(%[[S1_PTR]], %[[S1_LEN]], %[[S2_PTR]], %[[S2_LEN]], %[[BACK]]) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
! CHECK: %[[RES_I32:.*]] = fir.convert %[[RES_I64]] : (i64) -> i32
! CHECK: hlfir.assign %[[RES_I32]] to %[[RET_DECL]]#0 : i32, !fir.ref<i32>
! CHECK: %[[RES:.*]] = fir.load %[[RET_DECL]]#0 : !fir.ref<i32>
! CHECK: return %[[RES]] : i32
  character(*) :: s1, s2
  verify_test2 = verify(s1, s2, .true.)
end function verify_test2

! CHECK-LABEL: func @_QPtest_optional(
! CHECK-SAME:  %[[STRING_ARG:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK-SAME:  %[[SET_ARG:.*]]: !fir.boxchar<1>
! CHECK-SAME:  %[[BACK_ARG:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
subroutine test_optional(string, set, back)
  character (*) :: string(:), set
  logical, optional :: back(:)
! CHECK: %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[BACK_DECL:.*]]:2 = hlfir.declare %[[BACK_ARG]] dummy_scope %[[DSCOPE]] arg 3 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFtest_optionalEback"} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
! CHECK: %[[SET_UNBOX:.*]]:2 = fir.unboxchar %[[SET_ARG]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[SET_DECL:.*]]:2 = hlfir.declare %[[SET_UNBOX]]#0 typeparams %[[SET_UNBOX]]#1 dummy_scope %[[DSCOPE]] arg 2 {uniq_name = "_QFtest_optionalEset"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK: %[[STRING_DECL:.*]]:2 = hlfir.declare %[[STRING_ARG]] dummy_scope %[[DSCOPE]] arg 1 {uniq_name = "_QFtest_optionalEstring"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
  print *, verify(string, set, back)
! CHECK: %[[PRESENT:.*]] = fir.is_present %[[BACK_DECL]]#0 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> i1
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[DIMS:.*]]:3 = fir.box_dims %[[STRING_DECL]]#0, %[[C0]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
! CHECK: %[[SHAPE:.*]] = fir.shape %[[DIMS]]#1 : (index) -> !fir.shape<1>
! CHECK: %[[ELEMENTAL:.*]] = hlfir.elemental %[[SHAPE]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK: ^bb0(%[[I:.*]]: index):
! CHECK:   %[[CHAR_LEN:.*]] = fir.box_elesize %[[STRING_DECL]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:   %[[STRING_I:.*]] = hlfir.designate %[[STRING_DECL]]#0 (%[[I]])  typeparams %[[CHAR_LEN]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:   %[[UNBOX_I:.*]]:2 = fir.unboxchar %[[STRING_I]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:   %[[BACK_I_VAL:.*]] = fir.if %[[PRESENT]] -> (!fir.logical<4>) {
! CHECK:     %[[BACK_I_REF:.*]] = hlfir.designate %[[BACK_DECL]]#0 (%[[I]])  : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:     %[[VAL:.*]] = fir.load %[[BACK_I_REF]] : !fir.ref<!fir.logical<4>>
! CHECK:     fir.result %[[VAL]] : !fir.logical<4>
! CHECK:   } else {
! CHECK:     %[[FALSE:.*]] = arith.constant false
! CHECK:     %[[FALSE_LOG:.*]] = fir.convert %[[FALSE]] : (i1) -> !fir.logical<4>
! CHECK:     fir.result %[[FALSE_LOG]] : !fir.logical<4>
! CHECK:   }
! CHECK:   %[[S1_PTR:.*]] = fir.convert %[[UNBOX_I]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:   %[[S1_LEN:.*]] = fir.convert %[[CHAR_LEN]] : (index) -> i64
! CHECK:   %[[S2_PTR:.*]] = fir.convert %[[SET_DECL]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:   %[[S2_LEN:.*]] = fir.convert %[[SET_UNBOX]]#1 : (index) -> i64
! CHECK:   %[[BACK_I_BOOL:.*]] = fir.convert %[[BACK_I_VAL]] : (!fir.logical<4>) -> i1
! CHECK:   %[[RES_I64:.*]] = fir.call @_FortranAVerify1(%[[S1_PTR]], %[[S1_LEN]], %[[S2_PTR]], %[[S2_LEN]], %[[BACK_I_BOOL]]) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
! CHECK:   %[[RES_I32:.*]] = fir.convert %[[RES_I64]] : (i64) -> i32
! CHECK:   hlfir.yield_element %[[RES_I32]] : i32
! CHECK: }
end subroutine

! CHECK: func private @_FortranAVerify(
! CHECK: func private @_FortranAVerify1(
