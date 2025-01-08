// RUN: %flang_fc1 -emit-llvm -O1 -funroll-loops -mllvm -force-vector-width=1 -o- %s | FileCheck %s --check-prefixes=CHECK,UNROLL
// RUN: %flang_fc1 -emit-llvm -O2 -mllvm -force-vector-width=1 -o- %s | FileCheck %s --check-prefixes=CHECK,UNROLL
// RUN: %flang_fc1 -emit-llvm -O1 -fno-unroll-loops -mllvm -force-vector-width=1 -o- %s | FileCheck %s --check-prefixes=CHECK,NO-UNROLL
// RUN: %flang_fc1 -emit-llvm -O1 -mllvm -force-vector-width=1 -o- %s | FileCheck %s --check-prefixes=CHECK,NO-UNROLL

// CHECK-LABEL: @unroll
// CHECK-SAME: (ptr nocapture writeonly %[[ARG0:.*]])
func.func @unroll(%arg0: !fir.ref<!fir.array<1000xf64>> {fir.bindc_name = "a"}) {
  // CHECK:   %[[GEPIV:.*]] = getelementptr i8, ptr %0, i64 -8
  %scope = fir.dummy_scope : !fir.dscope
  %c1000 = arith.constant 1000 : index
  %shape = fir.shape %c1000 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%shape) dummy_scope %scope {uniq_name = "unrollEa"} : (!fir.ref<!fir.array<1000xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1000xf64>>, !fir.ref<!fir.array<1000xf64>>)
  %c1 = arith.constant 1 : index
  fir.do_loop %arg1 = %c1 to %c1000 step %c1 {
    // CHECK: [[BLK:.*]]:

    // NO-UNROLL-NEXT: %[[PHI:.*]] = phi i64 [ 1, %{{.*}} ], [ %[[NIV:.*]], %[[BLK]] ]
    // NO-UNROLL-NEXT: %[[IV_D:.*]] = uitofp nneg i64 %[[PHI]] to double
    // NO-UNROLL-NEXT: %[[GEP:.*]] = getelementptr double, ptr %[[GEPIV]], i64 %[[PHI]]
    // NO-UNROLL-NEXT: store double %[[IV_D]], ptr %[[GEP]]
    // NO-UNROLL-NEXT: %[[NIV:.*]] = add nuw nsw i64 %{{.*}}, 1
    // NO-UNROLL-NEXT: %[[EXIT:.*]] = icmp eq i64 %[[NIV]], 1001
    // NO-UNROLL-NEXT: br i1 %[[EXIT]], label %{{.*}}, label %[[BLK]]

    // UNROLL-NEXT: %[[PHI:.*]] = phi i64 [ 0, %{{.*}} ], [ %[[NIV:.*]], %[[BLK]] ]
    // UNROLL-NEXT: %[[IV0:.*]] = or disjoint i64 %[[PHI]], 1
    // UNROLL-NEXT: %[[IV1:.*]] = add i64 %[[PHI]], 2
    // UNROLL-NEXT: %[[IV0_D:.*]] = uitofp nneg i64 %[[IV0]] to double
    // UNROLL-NEXT: %[[IV1_D:.*]] = uitofp nneg i64 %[[IV1]] to double
    // UNROLL-NEXT: %[[GEP0:.*]] = getelementptr double, ptr %[[ARG0]], i64 %[[PHI]]
    // UNROLL-NEXT: %[[GEP1:.*]] = getelementptr double, ptr %[[GEPIV]], i64 %[[IV1]]
    // UNROLL-NEXT: store double %[[IV0_D]], ptr %[[GEP0]]
    // UNROLL-NEXT: store double %[[IV1_D]], ptr %[[GEP1]]
    // UNROLL-NEXT: %[[NIV:.*]] = add nuw i64 %[[PHI]], 2
    // UNROLL-NEXT: %[[EXIT:.*]] = icmp eq i64 %[[NIV]], 1000
    // UNROLL-NEXT: br i1 %[[EXIT]], label %{{.*}}, label %[[BLK]]
    %iv = fir.convert %arg1 : (index) -> f64
    %ai = hlfir.designate %a#0 (%arg1)  : (!fir.ref<!fir.array<1000xf64>>, index) -> !fir.ref<f64>
    hlfir.assign %iv to %ai : f64, !fir.ref<f64>
  }
  return
}
