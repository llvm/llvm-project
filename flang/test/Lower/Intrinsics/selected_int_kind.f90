! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPselected_int_kind_test1(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.ref<i8> {fir.bindc_name = "a"}) {
! CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} : (!fir.ref<i8>, !fir.dscope) -> (!fir.ref<i8>, !fir.ref<i8>)
! CHECK:         %[[KIND:.*]] = arith.constant 1 : i32
! CHECK:         %[[APTR:.*]] = fir.convert %[[DECL]]#0 : (!fir.ref<i8>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[RES:.*]] = fir.call @_FortranASelectedIntKind(%{{.*}}, %{{.*}}, %[[APTR]], %[[KIND]]) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         %[[CONV:.*]] = fir.convert %[[RES]] : (i32) -> i8
! CHECK:         hlfir.assign %[[CONV]] to %{{.*}} : i8, !fir.ref<i8>
! CHECK:         return
! CHECK:       }

subroutine selected_int_kind_test1(a)
  integer(1) :: a, res
  res = selected_int_kind(a)
end

! CHECK-LABEL: func.func @_QPselected_int_kind_test2(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.ref<i16> {fir.bindc_name = "a"}) {
! CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} : (!fir.ref<i16>, !fir.dscope) -> (!fir.ref<i16>, !fir.ref<i16>)
! CHECK:         %[[KIND:.*]] = arith.constant 2 : i32
! CHECK:         %[[APTR:.*]] = fir.convert %[[DECL]]#0 : (!fir.ref<i16>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[RES:.*]] = fir.call @_FortranASelectedIntKind(%{{.*}}, %{{.*}}, %[[APTR]], %[[KIND]]) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         %[[CONV:.*]] = fir.convert %[[RES]] : (i32) -> i16
! CHECK:         hlfir.assign %[[CONV]] to %{{.*}} : i16, !fir.ref<i16>
! CHECK:         return
! CHECK:       }

subroutine selected_int_kind_test2(a)
  integer(2) :: a, res
  res = selected_int_kind(a)
end

! CHECK-LABEL: func.func @_QPselected_int_kind_test4(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}) {
! CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[KIND:.*]] = arith.constant 4 : i32
! CHECK:         %[[APTR:.*]] = fir.convert %[[DECL]]#0 : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[RES:.*]] = fir.call @_FortranASelectedIntKind(%{{.*}}, %{{.*}}, %[[APTR]], %[[KIND]]) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         hlfir.assign %[[RES]] to %{{.*}} : i32, !fir.ref<i32>
! CHECK:         return
! CHECK:       }

subroutine selected_int_kind_test4(a)
  integer(4) :: a, res
  res = selected_int_kind(a)
end

! CHECK-LABEL: func.func @_QPselected_int_kind_test8(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "a"}) {
! CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:         %[[KIND:.*]] = arith.constant 8 : i32
! CHECK:         %[[APTR:.*]] = fir.convert %[[DECL]]#0 : (!fir.ref<i64>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[RES:.*]] = fir.call @_FortranASelectedIntKind(%{{.*}}, %{{.*}}, %[[APTR]], %[[KIND]]) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         %[[CONV:.*]] = fir.convert %[[RES]] : (i32) -> i64
! CHECK:         hlfir.assign %[[CONV]] to %{{.*}} : i64, !fir.ref<i64>
! CHECK:         return
! CHECK:       }

subroutine selected_int_kind_test8(a)
  integer(8) :: a, res
  res = selected_int_kind(a)
end

! CHECK-LABEL: func.func @_QPselected_int_kind_test16(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.ref<i128> {fir.bindc_name = "a"}) {
! CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} : (!fir.ref<i128>, !fir.dscope) -> (!fir.ref<i128>, !fir.ref<i128>)
! CHECK:         %[[KIND:.*]] = arith.constant 16 : i32
! CHECK:         %[[APTR:.*]] = fir.convert %[[DECL]]#0 : (!fir.ref<i128>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[RES:.*]] = fir.call @_FortranASelectedIntKind(%{{.*}}, %{{.*}}, %[[APTR]], %[[KIND]]) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         %[[CONV:.*]] = fir.convert %[[RES]] : (i32) -> i128
! CHECK:         hlfir.assign %[[CONV]] to %{{.*}} : i128, !fir.ref<i128>
! CHECK:         return
! CHECK:       }

subroutine selected_int_kind_test16(a)
  integer(16) :: a, res
  res = selected_int_kind(a)
end
