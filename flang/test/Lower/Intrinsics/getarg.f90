! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-32 -DDEFAULT_INTEGER_SIZE=32 %s
! RUN: %flang_fc1 -fdefault-integer-8 -emit-hlfir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-64 -DDEFAULT_INTEGER_SIZE=64 %s

! CHECK-LABEL: func @_QPgetarg_test(
! CHECK-SAME: %[[pos:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}, %[[value:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine getarg_test(pos, value)
integer :: pos
character(len=32) :: value
call getarg(pos, value)
! CHECK: %[[valueUnboxed:.*]]:2 = fir.unboxchar %[[value]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[valueCast:.*]] = fir.convert %[[valueUnboxed]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[valueCast]]
! CHECK: %[[posLoad:.*]] = fir.load {{.*}} : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK: %[[valueBoxed:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %[[absent:.*]] = fir.absent !fir.box<none>
! CHECK: %[[sourceFileString:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK: %[[sourceLine:.*]] = arith.constant [[# @LINE - 8]] : i32
! CHECK-64: %[[posCast:.*]] = fir.convert %[[posLoad]] : (i[[DEFAULT_INTEGER_SIZE]]) -> i32
! CHECK: %[[valueBoxedCast:.*]] = fir.convert %[[valueBoxed]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK-32: fir.call @_FortranAGetCommandArgument(%[[posLoad]], %[[valueBoxedCast]], %[[absent]], %[[absent]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (i32, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64: fir.call @_FortranAGetCommandArgument(%[[posCast]], %[[valueBoxedCast]], %[[absent]], %[[absent]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (i32, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
end subroutine getarg_test
