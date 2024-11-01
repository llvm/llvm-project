! RUN: bbc -emit-fir %s -opt-transpose=false -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -O0 %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtranspose_test(
! CHECK-SAME: %[[source:.*]]: !fir.ref<!fir.array<2x3xf32>>{{.*}}) {
subroutine transpose_test(mat)
! CHECK:  %[[resultDescr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xf32>>>
   real :: mat(2,3)
   call bar_transpose_test(transpose(mat))
! CHECK:  %[[sourceBox:.*]] = fir.embox %[[source]]({{.*}}) : (!fir.ref<!fir.array<2x3xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<2x3xf32>>
! CHECK:  %[[zeroArray:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xf32>
! CHECK:  %[[c0:.*]] = arith.constant 0 : index
! CHECK:  %[[shapeResult:.*]] = fir.shape %[[c0]], %[[c0]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[resultBox:.*]] = fir.embox %[[zeroArray]](%[[shapeResult]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
! CHECK:  fir.store %[[resultBox]] to %[[resultDescr]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:  %[[resultOpaque:.*]] = fir.convert %[[resultDescr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:  %[[sourceOpaque:.*]] = fir.convert %[[sourceBox]] : (!fir.box<!fir.array<2x3xf32>>) -> !fir.box<none>
! CHECK:  %{{.*}} = fir.call @_FortranATranspose(%[[resultOpaque]], %[[sourceOpaque]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:  %[[tmp1:.*]] = fir.load %[[resultDescr]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:  %[[tmp2:.*]] = fir.box_addr %[[tmp1]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
! CHECK:  %[[tmp3:.*]] = fir.convert %[[tmp2]] : (!fir.heap<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<3x2xf32>>
! CHECK:  fir.call @_QPbar_transpose_test(%[[tmp3]]) {{.*}}: (!fir.ref<!fir.array<3x2xf32>>) -> ()
! CHECK:  fir.freemem %[[tmp2]] : !fir.heap<!fir.array<?x?xf32>
end subroutine

