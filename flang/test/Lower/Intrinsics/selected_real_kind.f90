! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPselected_real_kind_test1(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.ref<i8> {fir.bindc_name = "p"},
! CHECK-SAME:                                         %[[VAL_1:.*]]: !fir.ref<i8> {fir.bindc_name = "r"},
! CHECK-SAME:                                         %[[VAL_2:.*]]: !fir.ref<i8> {fir.bindc_name = "d"}) {
! CHECK:         arith.constant 1 : i32
! CHECK:         fir.call @_FortranASelectedRealKind(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         %[[CONV:.*]] = fir.convert %{{.*}} : (i32) -> i8
! CHECK:         hlfir.assign %[[CONV]] to %{{.*}} : i8, !fir.ref<i8>
! CHECK:         return
! CHECK:       }

subroutine selected_real_kind_test1(p, r, d)
  integer(1) :: p, r, d, res
  res = selected_real_kind(P=p, R=r, RADIX=d)
end

! CHECK-LABEL: func.func @_QPselected_real_kind_test2(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.ref<i16> {fir.bindc_name = "p"},
! CHECK-SAME:                                         %[[VAL_1:.*]]: !fir.ref<i16> {fir.bindc_name = "r"},
! CHECK-SAME:                                         %[[VAL_2:.*]]: !fir.ref<i16> {fir.bindc_name = "d"}) {
! CHECK:         arith.constant 2 : i32
! CHECK:         fir.call @_FortranASelectedRealKind(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         %[[CONV:.*]] = fir.convert %{{.*}} : (i32) -> i16
! CHECK:         hlfir.assign %[[CONV]] to %{{.*}} : i16, !fir.ref<i16>
! CHECK:         return
! CHECK:       }

subroutine selected_real_kind_test2(p, r, d)
  integer(2) :: p, r, d, res
  res = selected_real_kind(P=p, R=r, RADIX=d)
end

! CHECK-LABEL: func.func @_QPselected_real_kind_test4(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "p"},
! CHECK-SAME:                                         %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "r"},
! CHECK-SAME:                                         %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "d"}) {
! CHECK:         arith.constant 4 : i32
! CHECK:         fir.call @_FortranASelectedRealKind(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! CHECK:         return
! CHECK:       }

subroutine selected_real_kind_test4(p, r, d)
  integer(4) :: p, r, d, res
  res = selected_real_kind(P=p, R=r, RADIX=d)
end

! CHECK-LABEL: func.func @_QPselected_real_kind_test8(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "p"},
! CHECK-SAME:                                         %[[VAL_1:.*]]: !fir.ref<i64> {fir.bindc_name = "r"},
! CHECK-SAME:                                         %[[VAL_2:.*]]: !fir.ref<i64> {fir.bindc_name = "d"}) {
! CHECK:         arith.constant 8 : i32
! CHECK:         fir.call @_FortranASelectedRealKind(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         %[[CONV:.*]] = fir.convert %{{.*}} : (i32) -> i64
! CHECK:         hlfir.assign %[[CONV]] to %{{.*}} : i64, !fir.ref<i64>
! CHECK:         return
! CHECK:       }

subroutine selected_real_kind_test8(p, r, d)
  integer(8) :: p, r, d, res
  res = selected_real_kind(P=p, R=r, RADIX=d)
end

! CHECK-LABEL: func.func @_QPselected_real_kind_test16(
! CHECK-SAME:                                          %[[VAL_0:.*]]: !fir.ref<i128> {fir.bindc_name = "p"},
! CHECK-SAME:                                          %[[VAL_1:.*]]: !fir.ref<i128> {fir.bindc_name = "r"},
! CHECK-SAME:                                          %[[VAL_2:.*]]: !fir.ref<i128> {fir.bindc_name = "d"}) {
! CHECK:         arith.constant 16 : i32
! CHECK:         fir.call @_FortranASelectedRealKind(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         %[[CONV:.*]] = fir.convert %{{.*}} : (i32) -> i128
! CHECK:         hlfir.assign %[[CONV]] to %{{.*}} : i128, !fir.ref<i128>
! CHECK:         return
! CHECK:       }

subroutine selected_real_kind_test16(p, r, d)
  integer(16) :: p, r, d, res
  res = selected_real_kind(P=p, R=r, RADIX=d)
end

! CHECK-LABEL: func.func @_QPselected_real_kind_test_rd(
! CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "r"},
! CHECK-SAME:                                           %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "d"}) {
! CHECK:         fir.absent !fir.ref<i1>
! CHECK:         arith.constant 0 : i32
! CHECK:         fir.call @_FortranASelectedRealKind(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! CHECK:         return
! CHECK:       }

subroutine selected_real_kind_test_rd(r, d)
  integer :: r, d, res
  res = selected_real_kind(R=r, RADIX=d)
end

! CHECK-LABEL: func.func @_QPselected_real_kind_test_pd(
! CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "p"},
! CHECK-SAME:                                           %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "d"}) {
! CHECK:         fir.absent !fir.ref<i1>
! CHECK:         arith.constant 0 : i32
! CHECK:         fir.call @_FortranASelectedRealKind(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! CHECK:         return
! CHECK:       }

subroutine selected_real_kind_test_pd(p, d)
  integer :: p, d, res
  res = selected_real_kind(P=p, RADIX=d)
end

! CHECK-LABEL: func.func @_QPselected_real_kind_test_pr(
! CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "p"},
! CHECK-SAME:                                           %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "r"}) {
! CHECK:         fir.absent !fir.ref<i1>
! CHECK:         arith.constant 0 : i32
! CHECK:         fir.call @_FortranASelectedRealKind(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! CHECK:         return
! CHECK:       }

subroutine selected_real_kind_test_pr(p, r)
  integer :: p, r, res
  res = selected_real_kind(P=p, R=r)
end
