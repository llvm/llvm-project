! RUN: bbc -emit-fir %s -o - | fir-opt --canonicalize | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | fir-opt --canonicalize | FileCheck %s

! CHECK-LABEL: test1
! CHECK-SAME: (%[[XREF:.*]]: !fir.ref<i32> {{.*}}, %[[CBOX:.*]]: !fir.boxchar<1> {{.*}})
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[FALSE:.*]] = arith.constant false
! CHECK: %[[TEMP:.*]] = fir.alloca !fir.char<1> {adapt.valuebyref}
! CHECK: %[[C:.*]]:2 = fir.unboxchar %[[CBOX]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[X:.*]] = fir.load %[[XREF]] : !fir.ref<i32>
! CHECK: %[[X_I8:.*]] = fir.convert %[[X]] : (i32) -> i8
! CHECK: %[[UNDEF:.*]] = fir.undefined !fir.char<1>
! CHECK: %[[XCHAR:.*]] = fir.insert_value %[[UNDEF]], %[[X_I8]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK: fir.store %[[XCHAR]] to %[[TEMP]] : !fir.ref<!fir.char<1>>
! CHECK: %[[C1_I64:.*]] = fir.convert %[[C1]] : (index) -> i64
! CHECK: %[[C_CVT:.*]] = fir.convert %[[C]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK: %[[TEMP_WITH_XCHAR:.*]] = fir.convert %[[TEMP]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
! CHECK: fir.call @llvm.memmove.p0i8.p0i8.i64(%[[C_CVT]], %[[TEMP_WITH_XCHAR]], %[[C1_I64]], %[[FALSE]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
subroutine test1(x, c)
  integer :: x
  character :: c
  c = achar(x)
end subroutine
