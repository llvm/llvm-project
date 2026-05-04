! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPadjustr_test() {
subroutine adjustr_test
    character(len=12) :: adjust_str = '0123456789  '
  ! CHECK: %[[strBox:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,12>>>
  ! CHECK: %[[addr0:.*]] = fir.address_of(@_QFadjustr_testEadjust_str) : !fir.ref<!fir.char<1,12>>
  ! CHECK: %[[c12:.*]] = arith.constant 12 : index
  ! CHECK: %[[decl:.*]]:2 = hlfir.declare %[[addr0]] typeparams %[[c12]] {{.*}} : (!fir.ref<!fir.char<1,12>>, index) -> (!fir.ref<!fir.char<1,12>>, !fir.ref<!fir.char<1,12>>)
  ! CHECK: %[[eBox:.*]] = fir.embox %[[decl]]#0 : (!fir.ref<!fir.char<1,12>>) -> !fir.box<!fir.char<1,12>>
  ! CHECK: %[[r0:.*]] = fir.zero_bits !fir.heap<!fir.char<1,12>>
  ! CHECK: %[[r1:.*]] = fir.embox %[[r0]] : (!fir.heap<!fir.char<1,12>>) -> !fir.box<!fir.heap<!fir.char<1,12>>>
  ! CHECK: fir.store %[[r1]] to %[[strBox]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,12>>>>
  ! CHECK: %[[r2:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
  ! CHECK: %[[r3:.*]] = fir.convert %[[strBox]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,12>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[r4:.*]] = fir.convert %[[eBox]] : (!fir.box<!fir.char<1,12>>) -> !fir.box<none>
  ! CHECK: %[[r5:.*]] = fir.convert %[[r2]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK: fir.call @_FortranAAdjustr(%[[r3]], %[[r4]], %[[r5]], %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
    adjust_str = adjustr(adjust_str)
  end subroutine
