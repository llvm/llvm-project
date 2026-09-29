! Check that a box is created instead of a temp to write to a char array.
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine io_char_array
  character(12) :: r(2) = 'badbadbadbad'
  write(r(1:2),8)
  print *, r
1 format((1X,A))
8 format("new"/"data")
end subroutine

! CHECK-LABEL: func.func @_QPio_char_array()
! CHECK: %[[R_ADDR:.*]] = fir.address_of(@_QFio_char_arrayEr) : !fir.ref<!fir.array<2x!fir.char<1,12>>>
! CHECK: %[[C12:.*]] = arith.constant 12 : index
! CHECK: %[[C2:.*]] = arith.constant 2 : index
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C2]] : (index) -> !fir.shape<1>
! CHECK: %[[R_DECL:.*]]:2 = hlfir.declare %[[R_ADDR]](%[[SHAPE]]) typeparams %[[C12]] {uniq_name = "_QFio_char_arrayEr"} : (!fir.ref<!fir.array<2x!fir.char<1,12>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<2x!fir.char<1,12>>>, !fir.ref<!fir.array<2x!fir.char<1,12>>>)
! CHECK: %[[C1_1:.*]] = arith.constant 1 : index
! CHECK: %[[C2_1:.*]] = arith.constant 2 : index
! CHECK: %[[C1_2:.*]] = arith.constant 1 : index
! CHECK: %[[C2_2:.*]] = arith.constant 2 : index
! CHECK: %[[SHAPE2:.*]] = fir.shape %[[C2_2]] : (index) -> !fir.shape<1>
! CHECK: %[[DESIGNATE:.*]] = hlfir.designate %[[R_DECL]]#0 (%[[C1_1]]:%[[C2_1]]:%[[C1_2]])  shape %[[SHAPE2]] typeparams %[[C12]] : (!fir.ref<!fir.array<2x!fir.char<1,12>>>, index, index, index, !fir.shape<1>, index) -> !fir.ref<!fir.array<2x!fir.char<1,12>>>
! CHECK: %[[SHAPE3:.*]] = fir.shape %[[C2_2]] : (index) -> !fir.shape<1>
! CHECK: %[[EMBOX:.*]] = fir.embox %[[DESIGNATE]](%[[SHAPE3]]) : (!fir.ref<!fir.array<2x!fir.char<1,12>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1,12>>>
! CHECK: %[[EMBOX_NONE:.*]] = fir.convert %[[EMBOX]] : (!fir.box<!fir.array<2x!fir.char<1,12>>>) -> !fir.box<none>
! CHECK: %[[FORMAT:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK: %[[FORMAT_PTR:.*]] = fir.convert %[[FORMAT]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK: %{{.*}} = fir.call @_FortranAioBeginInternalArrayFormattedOutput(%[[EMBOX_NONE]], %[[FORMAT_PTR]], {{.*}}) {{.*}}: (!fir.box<none>, !fir.ref<i8>, i64, !fir.box<none>, !fir.ref<!fir.llvm_ptr<i8>>, i64, !fir.ref<i8>, i32) -> !fir.ref<i8>
