// DEFINE: %{triple} =
// DEFINE: %{check-unroll} = %flang_fc1 -emit-llvm -O1 -funroll-loops -mllvm -force-vector-width=2 -triple %{triple} -o- %s | FileCheck %s --check-prefixes=CHECK,UNROLL
// DEFINE: %{check-nounroll} = %flang_fc1 -emit-llvm -O1 -mllvm -force-vector-width=2 -triple %{triple} -o- %s | FileCheck %s --check-prefixes=CHECK,NO-UNROLL

// REDEFINE: %{triple} = aarch64-unknown-linux-gnu
// RUN: %if aarch64-registered-target %{ %{check-unroll} %}
// RUN: %if aarch64-registered-target %{ %{check-nounroll} %}

// REDEFINE: %{triple} = x86_64-unknown-linux-gnu
// RUN: %if x86-registered-target %{ %{check-unroll} %}
// RUN: %if x86-registered-target %{ %{check-nounroll} %}

// CHECK-LABEL: @unroll
// CHECK-SAME: (ptr nocapture writeonly %[[ARG0:.*]])
func.func @unroll(%arg0: !fir.ref<!fir.array<1000 x index>> {fir.bindc_name = "a"}) {
  %scope = fir.dummy_scope : !fir.dscope
  %c1000 = arith.constant 1000 : index
  %shape = fir.shape %c1000 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%shape) dummy_scope %scope {uniq_name = "unrollEa"} : (!fir.ref<!fir.array<1000xindex>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1000 x index>>, !fir.ref<!fir.array<1000 x index>>)
  %c1 = arith.constant 1 : index
  fir.do_loop %arg1 = %c1 to %c1000 step %c1 {
    // CHECK: br label %[[BLK:.*]]
    // CHECK: [[BLK]]:
    // CHECK-NEXT: %[[IND:.*]] = phi i64 [ 0, %{{.*}} ], [ %[[NIV:.*]], %[[BLK]] ]
    // CHECK-NEXT: %[[VIND:.*]] = phi <2 x i64> [ <i64 1, i64 2>, %{{.*}} ], [ %[[NVIND:.*]], %[[BLK]] ]

    // NO-UNROLL-NEXT: %[[GEP:.*]] = getelementptr i64, ptr %[[ARG0]], i64 %[[IND]]
    // NO-UNROLL-NEXT: store <2 x i64> %[[VIND]], ptr %[[GEP]]
    // NO-UNROLL-NEXT: %[[NIV:.*]] = add nuw i64 %{{.*}}, 2
    // NO-UNROLL-NEXT: %[[NVIND]] = add <2 x i64> %[[VIND]], splat (i64 2)

    // UNROLL-NEXT: %[[VIND1:.*]] = add <2 x i64> %[[VIND]], splat (i64 2)
    // UNROLL-NEXT: %[[GEP0:.*]] = getelementptr i64, ptr %[[ARG0]], i64 %[[IND]]
    // UNROLL-NEXT: %[[GEP1:.*]] = getelementptr i8, ptr %[[GEP0]], i64 16
    // UNROLL-NEXT: store <2 x i64> %[[VIND]], ptr %[[GEP0]]
    // UNROLL-NEXT: store <2 x i64> %[[VIND1]], ptr %[[GEP1]]
    // UNROLL-NEXT: %[[NIV:.*]] = add nuw i64 %[[IND]], 4
    // UNROLL-NEXT: %[[NVIND:.*]] = add <2 x i64> %[[VIND]], splat (i64 4)

    // CHECK-NEXT: %[[EXIT:.*]] = icmp eq i64 %[[NIV]], 1000
    // CHECK-NEXT: br i1 %[[EXIT]], label %{{.*}}, label %[[BLK]]
    %ai = hlfir.designate %a#0 (%arg1)  : (!fir.ref<!fir.array<1000 x index>>, index) -> !fir.ref<index>
    hlfir.assign %arg1 to %ai : index, !fir.ref<index>
  }
  return
}
