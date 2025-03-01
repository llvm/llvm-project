// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Test __atomic_* built-ins that have a memory order parameter with a runtime
// value.  This requires generating a switch statement, so the amount of
// generated code is surprisingly large.
//
// Only a representative sample of atomic operations are tested: one read-only
// operation (atomic_load), one write-only operation (atomic_store), one
// read-write operation (atomic_exchange), and the most complex operation
// (atomic_compare_exchange).

int runtime_load(int *ptr, int order) {
  return __atomic_load_n(ptr, order);
}

// CHECK: %[[ptr:.*]] = cir.load %[[ptr_var:.*]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: %[[order:.*]] = cir.load %[[order_var:.*]] : !cir.ptr<!s32i>, !s32i
// CHECK: cir.switch (%[[order]] : !s32i) {
// CHECK: cir.case(default, []) {
// CHECK:   %[[T8:.*]] = cir.load atomic(relaxed) %[[ptr]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.store %[[T8]], %[[temp_var:.*]] : !s32i, !cir.ptr<!s32i>
// CHECK:   cir.break
// CHECK: }
// CHECK: cir.case(anyof, [#cir.int<1> : !s32i, #cir.int<2> : !s32i]) {
// CHECK:   %[[T8:.*]] = cir.load atomic(acquire) %[[ptr]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.store %[[T8]], %[[temp_var]] : !s32i, !cir.ptr<!s32i>
// CHECK:   cir.break
// CHECK: }
// CHECK: cir.case(equal, [#cir.int<5> : !s32i]) {
// CHECK:   %[[T8:.*]] = cir.load atomic(seq_cst) %[[ptr]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.store %[[T8]], %[[temp_var]] : !s32i, !cir.ptr<!s32i>
// CHECK:   cir.break
// CHECK: }
// CHECK: }

void atomic_store_n(int* ptr, int val, int order) {
  __atomic_store_n(ptr, val, order);
}

// CHECK: %[[ptr:.*]] = cir.load %[[ptr_var:.*]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: %[[order:.*]] = cir.load %[[order_var:.*]] : !cir.ptr<!s32i>, !s32i
// CHECK: %[[val:.*]] = cir.load %[[val_var:.*]] : !cir.ptr<!s32i>, !s32i
// CHECK: cir.store %[[val]], %[[temp_var:.*]] : !s32i, !cir.ptr<!s32i>
// CHECK: cir.switch (%[[order]] : !s32i) {
// CHECK: cir.case(default, []) {
// CHECK:   %[[T7:.*]] = cir.load %[[temp_var:.*]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.store atomic(relaxed) %[[T7]], %[[ptr]] : !s32i, !cir.ptr<!s32i>
// CHECK:   cir.break
// CHECK: }
// CHECK: cir.case(equal, [#cir.int<3> : !s32i]) {
// CHECK:   %[[T7:.*]] = cir.load %[[temp_var:.*]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.store atomic(release) %[[T7]], %[[ptr]] : !s32i, !cir.ptr<!s32i>
// CHECK:   cir.break
// CHECK: }
// CHECK: cir.case(equal, [#cir.int<5> : !s32i]) {
// CHECK:   %[[T7:.*]] = cir.load %[[temp_var:.*]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.store atomic(seq_cst) %[[T7]], %[[ptr]] : !s32i, !cir.ptr<!s32i>
// CHECK:   cir.break
// CHECK: }
// CHECK: }

int atomic_exchange_n(int* ptr, int val, int order) {
  return __atomic_exchange_n(ptr, val, order);
}

// CHECK: %[[ptr:.*]] = cir.load %[[ptr_var:.*]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: %[[order:.*]] = cir.load %[[order_var:.*]] : !cir.ptr<!s32i>, !s32i
// CHECK: %[[val:.*]] = cir.load %[[val_var:.*]] : !cir.ptr<!s32i>, !s32i
// CHECK: cir.store %[[val]], %[[temp_var:.*]] : !s32i, !cir.ptr<!s32i>
// CHECK: cir.switch (%[[order]] : !s32i) {
// CHECK: cir.case(default, []) {
// CHECK:   %[[T11:.*]] = cir.load %[[temp_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:   %[[T12:.*]] = cir.atomic.xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[T11]] : !s32i, relaxed) : !s32i
// CHECK:   cir.store %[[T12]], %[[result:.*]] : !s32i, !cir.ptr<!s32i>
// CHECK:   cir.break
// CHECK: }
// CHECK: cir.case(anyof, [#cir.int<1> : !s32i, #cir.int<2> : !s32i]) {
// CHECK:   %[[T11:.*]] = cir.load %[[temp_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:   %[[T12:.*]] = cir.atomic.xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[T11]] : !s32i, acquire) : !s32i
// CHECK:   cir.store %[[T12]], %[[result]] : !s32i, !cir.ptr<!s32i>
// CHECK:   cir.break
// CHECK: }
// CHECK: cir.case(equal, [#cir.int<3> : !s32i]) {
// CHECK:   %[[T11:.*]] = cir.load %[[temp_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:   %[[T12:.*]] = cir.atomic.xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[T11]] : !s32i, release) : !s32i
// CHECK:   cir.store %[[T12]], %[[result]] : !s32i, !cir.ptr<!s32i>
// CHECK:   cir.break
// CHECK: }
// CHECK: cir.case(equal, [#cir.int<4> : !s32i]) {
// CHECK:   %[[T11:.*]] = cir.load %[[temp_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:   %[[T12:.*]] = cir.atomic.xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[T11]] : !s32i, acq_rel) : !s32i
// CHECK:   cir.store %[[T12]], %[[result]] : !s32i, !cir.ptr<!s32i>
// CHECK:   cir.break
// CHECK: }
// CHECK: cir.case(equal, [#cir.int<5> : !s32i]) {
// CHECK:   %[[T11:.*]] = cir.load %[[temp_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:   %[[T12:.*]] = cir.atomic.xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[T11]] : !s32i, seq_cst) : !s32i
// CHECK:   cir.store %[[T12]], %[[result]] : !s32i, !cir.ptr<!s32i>
// CHECK:   cir.break
// CHECK: }
// CHECK: }

bool atomic_compare_exchange_n(int* ptr, int* expected,
                               int desired, int success, int failure) {
  return __atomic_compare_exchange_n(ptr, expected, desired, false,
                                     success, failure);
}

// CHECK: %[[ptr:.*]] = cir.load %[[T0:.*]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: %[[success:.*]] = cir.load %[[T3:.*]] : !cir.ptr<!s32i>, !s32i
// CHECK: %[[expected_addr:.*]] = cir.load %[[T1:.*]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: %[[T11:.*]] = cir.load %[[T2:.*]] : !cir.ptr<!s32i>, !s32i
// CHECK: cir.store %[[T11]], %[[desired_var:.*]] : !s32i, !cir.ptr<!s32i>
// CHECK: %[[failure:.*]] = cir.load %[[T4:.*]] : !cir.ptr<!s32i>, !s32i
// CHECK: %[[T13:.*]] = cir.const #false
// CHECK: cir.switch (%[[success]] : !s32i) {
// CHECK: cir.case(default, []) {
// CHECK:   cir.switch (%[[failure]] : !s32i) {
// CHECK:   cir.case(default, []) {
// CHECK:     %[[expected:.*]] = cir.load %[[expected_addr]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[desired:.*]] = cir.load %[[desired_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %old, %cmp = cir.atomic.cmp_xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[expected]] : !s32i, %[[desired]] : !s32i, success = relaxed, failure = relaxed) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK:     %[[succeeded:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK:     cir.if %[[succeeded]] {
// CHECK:       cir.store %old, %[[expected_addr]] : !s32i, !cir.ptr<!s32i>
// CHECK:     }
// CHECK:     cir.store %cmp, %[[result_var:.*]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.break
// CHECK:   }
// CHECK:   cir.case(anyof, [#cir.int<1> : !s32i, #cir.int<2> : !s32i]) {
// CHECK:     %[[expected:.*]] = cir.load %[[expected_addr]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[desired:.*]] = cir.load %[[desired_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %old, %cmp = cir.atomic.cmp_xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[expected]] : !s32i, %[[desired]] : !s32i, success = relaxed, failure = acquire) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK:     %[[succeeded:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK:     cir.if %[[succeeded]] {
// CHECK:       cir.store %old, %[[expected_addr]] : !s32i, !cir.ptr<!s32i>
// CHECK:     }
// CHECK:     cir.store %cmp, %[[result_var]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.break
// CHECK:   }
// CHECK:   cir.case(equal, [#cir.int<5> : !s32i]) {
// CHECK:     %[[expected:.*]] = cir.load %[[expected_addr]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[desired:.*]] = cir.load %[[desired_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %old, %cmp = cir.atomic.cmp_xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[expected]] : !s32i, %[[desired]] : !s32i, success = relaxed, failure = seq_cst) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK:     %[[succeeded:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK:     cir.if %[[succeeded]] {
// CHECK:       cir.store %old, %[[expected_addr]] : !s32i, !cir.ptr<!s32i>
// CHECK:     }
// CHECK:     cir.store %cmp, %[[result_var]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.break
// CHECK:   }
// CHECK:   }
// CHECK:   cir.break
// CHECK: }
// CHECK: cir.case(anyof, [#cir.int<1> : !s32i, #cir.int<2> : !s32i]) {
// CHECK:   cir.switch (%[[failure]] : !s32i) {
// CHECK:   cir.case(default, []) {
// CHECK:     %[[expected:.*]] = cir.load %[[expected_addr]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[desired:.*]] = cir.load %[[desired_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %old, %cmp = cir.atomic.cmp_xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[expected]] : !s32i, %[[desired]] : !s32i, success = acquire, failure = relaxed) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK:     %[[succeeded:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK:     cir.if %[[succeeded]] {
// CHECK:       cir.store %old, %[[expected_addr]] : !s32i, !cir.ptr<!s32i>
// CHECK:     }
// CHECK:     cir.store %cmp, %[[result_var]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.break
// CHECK:   }
// CHECK:   cir.case(anyof, [#cir.int<1> : !s32i, #cir.int<2> : !s32i]) {
// CHECK:     %[[expected:.*]] = cir.load %[[expected_addr]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[desired:.*]] = cir.load %[[desired_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %old, %cmp = cir.atomic.cmp_xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[expected]] : !s32i, %[[desired]] : !s32i, success = acquire, failure = acquire) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK:     %[[succeeded:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK:     cir.if %[[succeeded]] {
// CHECK:       cir.store %old, %[[expected_addr]] : !s32i, !cir.ptr<!s32i>
// CHECK:     }
// CHECK:     cir.store %cmp, %[[result_var]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.break
// CHECK:   }
// CHECK:   cir.case(equal, [#cir.int<5> : !s32i]) {
// CHECK:     %[[expected:.*]] = cir.load %[[expected_addr]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[desired:.*]] = cir.load %[[desired_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %old, %cmp = cir.atomic.cmp_xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[expected]] : !s32i, %[[desired]] : !s32i, success = acquire, failure = seq_cst) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK:     %[[succeeded:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK:     cir.if %[[succeeded]] {
// CHECK:       cir.store %old, %[[expected_addr]] : !s32i, !cir.ptr<!s32i>
// CHECK:     }
// CHECK:     cir.store %cmp, %[[result_var]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.break
// CHECK:   }
// CHECK:   }
// CHECK:   cir.break
// CHECK: }
// CHECK: cir.case(equal, [#cir.int<3> : !s32i])
// CHECK:   cir.switch (%[[failure]] : !s32i) {
// CHECK:   cir.case(default, []) {
// CHECK:     %[[expected:.*]] = cir.load %[[expected_addr]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[desired:.*]] = cir.load %[[desired_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %old, %cmp = cir.atomic.cmp_xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[expected]] : !s32i, %[[desired]] : !s32i, success = release, failure = relaxed) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK:     %[[succeeded:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK:     cir.if %[[succeeded]] {
// CHECK:       cir.store %old, %[[expected_addr]] : !s32i, !cir.ptr<!s32i>
// CHECK:     }
// CHECK:     cir.store %cmp, %[[result_var]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.break
// CHECK:   }
// CHECK:   cir.case(anyof, [#cir.int<1> : !s32i, #cir.int<2> : !s32i]) {
// CHECK:     %[[expected:.*]] = cir.load %[[expected_addr]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[desired:.*]] = cir.load %[[desired_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %old, %cmp = cir.atomic.cmp_xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[expected]] : !s32i, %[[desired]] : !s32i, success = release, failure = acquire) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK:     %[[succeeded:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK:     cir.if %[[succeeded]] {
// CHECK:       cir.store %old, %[[expected_addr]] : !s32i, !cir.ptr<!s32i>
// CHECK:     }
// CHECK:     cir.store %cmp, %[[result_var]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.break
// CHECK:   }
// CHECK:   cir.case(equal, [#cir.int<5> : !s32i]) {
// CHECK:     %[[expected:.*]] = cir.load %[[expected_addr]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[desired:.*]] = cir.load %[[desired_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %old, %cmp = cir.atomic.cmp_xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[expected]] : !s32i, %[[desired]] : !s32i, success = release, failure = seq_cst) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK:     %[[succeeded:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK:     cir.if %[[succeeded]] {
// CHECK:       cir.store %old, %[[expected_addr]] : !s32i, !cir.ptr<!s32i>
// CHECK:     }
// CHECK:     cir.store %cmp, %[[result_var]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.break
// CHECK:   }
// CHECK:   }
// CHECK:   cir.break
// CHECK: }
// CHECK: cir.case(equal, [#cir.int<4> : !s32i]) {
// CHECK:   cir.switch (%[[failure]] : !s32i) {
// CHECK:   cir.case(default, []) {
// CHECK:     %[[expected:.*]] = cir.load %[[expected_addr]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[desired:.*]] = cir.load %[[desired_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %old, %cmp = cir.atomic.cmp_xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[expected]] : !s32i, %[[desired]] : !s32i, success = acq_rel, failure = relaxed) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK:     %[[succeeded:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK:     cir.if %[[succeeded]] {
// CHECK:       cir.store %old, %[[expected_addr]] : !s32i, !cir.ptr<!s32i>
// CHECK:     }
// CHECK:     cir.store %cmp, %[[result_var]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.break
// CHECK:   }
// CHECK:   cir.case(anyof, [#cir.int<1> : !s32i, #cir.int<2> : !s32i]) {
// CHECK:     %[[expected:.*]] = cir.load %[[expected_addr]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[desired:.*]] = cir.load %[[desired_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %old, %cmp = cir.atomic.cmp_xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[expected]] : !s32i, %[[desired]] : !s32i, success = acq_rel, failure = acquire) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK:     %[[succeeded:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK:     cir.if %[[succeeded]] {
// CHECK:       cir.store %old, %[[expected_addr]] : !s32i, !cir.ptr<!s32i>
// CHECK:     }
// CHECK:     cir.store %cmp, %[[result_var]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.break
// CHECK:   }
// CHECK:   cir.case(equal, [#cir.int<5> : !s32i]) {
// CHECK:     %[[expected:.*]] = cir.load %[[expected_addr]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[desired:.*]] = cir.load %[[desired_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %old, %cmp = cir.atomic.cmp_xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[expected]] : !s32i, %[[desired]] : !s32i, success = acq_rel, failure = seq_cst) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK:     %[[succeeded:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK:     cir.if %[[succeeded]] {
// CHECK:       cir.store %old, %[[expected_addr]] : !s32i, !cir.ptr<!s32i>
// CHECK:     }
// CHECK:     cir.store %cmp, %[[result_var]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.break
// CHECK:   }
// CHECK:   }
// CHECK:   cir.break
// CHECK: }
// CHECK: cir.case(equal, [#cir.int<5> : !s32i]) {
// CHECK:   cir.switch (%[[failure]] : !s32i) {
// CHECK:   cir.case(default, []) {
// CHECK:     %[[expected:.*]] = cir.load %[[expected_addr]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[desired:.*]] = cir.load %[[desired_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %old, %cmp = cir.atomic.cmp_xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[expected]] : !s32i, %[[desired]] : !s32i, success = seq_cst, failure = relaxed) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK:     %[[succeeded:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK:     cir.if %[[succeeded]] {
// CHECK:       cir.store %old, %[[expected_addr]] : !s32i, !cir.ptr<!s32i>
// CHECK:     }
// CHECK:     cir.store %cmp, %[[result_var]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.break
// CHECK:   }
// CHECK:   cir.case(anyof, [#cir.int<1> : !s32i, #cir.int<2> : !s32i]) {
// CHECK:     %[[expected:.*]] = cir.load %[[expected_addr]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[desired:.*]] = cir.load %[[desired_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %old, %cmp = cir.atomic.cmp_xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[expected]] : !s32i, %[[desired]] : !s32i, success = seq_cst, failure = acquire) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK:     %[[succeeded:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK:     cir.if %[[succeeded]] {
// CHECK:       cir.store %old, %[[expected_addr]] : !s32i, !cir.ptr<!s32i>
// CHECK:     }
// CHECK:     cir.store %cmp, %[[result_var]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.break
// CHECK:   }
// CHECK:   cir.case(equal, [#cir.int<5> : !s32i]) {
// CHECK:     %[[expected:.*]] = cir.load %[[expected_addr]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[desired:.*]] = cir.load %[[desired_var]] : !cir.ptr<!s32i>, !s32i
// CHECK:     %old, %cmp = cir.atomic.cmp_xchg(%[[ptr]] : !cir.ptr<!s32i>, %[[expected]] : !s32i, %[[desired]] : !s32i, success = seq_cst, failure = seq_cst) syncscope(system) align(4) : (!s32i, !cir.bool)
// CHECK:     %[[succeeded:.*]] = cir.unary(not, %cmp) : !cir.bool, !cir.bool
// CHECK:     cir.if %[[succeeded]] {
// CHECK:       cir.store %old, %[[expected_addr]] : !s32i, !cir.ptr<!s32i>
// CHECK:     }
// CHECK:     cir.store %cmp, %[[result_var]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.break
// CHECK:   }
// CHECK:   }
// CHECK:   cir.break
// CHECK: }
// CHECK: }

