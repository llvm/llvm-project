// RUN: %clang_cc1 -fopenacc -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:     -fclangir-call-conv-lowering -emit-cir %s -o - | FileCheck %s

struct Triple {
  int a, b, c;
};

struct HasCoercedCtor {
  HasCoercedCtor(Triple t = Triple{});
  ~HasCoercedCtor();
};

void privatized() {
  HasCoercedCtor c;
#pragma acc parallel private(c)
  ;
}

// A call outside a cir.func gets its coercion slot at the recipe's start.
// CHECK:      acc.private.recipe @privatization__ZTS14HasCoercedCtor : !cir.ptr<!rec_HasCoercedCtor> init {
// CHECK-NEXT:   ^bb0(%{{[^,)]+}}: !cir.ptr<!rec_HasCoercedCtor>
// CHECK-NEXT:   %[[COERCE:.*]] = cir.alloca "coerce" {{.*}} : !cir.ptr<!rec_anon_struct>
// CHECK-NEXT:   cir.alloca "openacc.private.init"
// CHECK-NEXT:   %[[TMP:.*]] = cir.alloca "agg.tmp{{[0-9]*}}" align(4) : !cir.ptr<!rec_Triple>
// CHECK:        %[[VIEW:.*]] = cir.cast bitcast %[[COERCE]] : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!rec_Triple>
// CHECK-NEXT:   cir.copy %[[TMP]] align(4) to %[[VIEW]] align(8) : !cir.ptr<!rec_Triple>
// CHECK-NEXT:   %[[A_PTR:.*]] = cir.get_member %[[COERCE]][0] {{.*}} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u64i>
// CHECK-NEXT:   %[[A:.*]] = cir.load %[[A_PTR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT:   %[[B_PTR:.*]] = cir.get_member %[[COERCE]][1] {{.*}} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!s32i>
// CHECK-NEXT:   %[[B:.*]] = cir.load %[[B_PTR]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT:   cir.call @_ZN14HasCoercedCtorC1E6Triple(%{{.+}}, %[[A]], %[[B]]) : (!cir.ptr<!rec_HasCoercedCtor> {{.*}}, !u64i, !s32i) -> ()
