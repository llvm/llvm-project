// RUN: %clang_cc1 -fopenacc -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:     -fclangir-call-conv-lowering -emit-cir %s -o - | FileCheck %s

struct Pair {
  long a, b;
};

// Pair is two eightbytes, so the constructor's argument is coerced and the
// call needs a coercion slot.  The recipe body is not inside a cir.func when
// CallConvLowering runs, so the slot goes at the start of the init region.
struct HasCoercedCtor {
  HasCoercedCtor(Pair p = Pair{});
  ~HasCoercedCtor();
};

void privatized() {
  HasCoercedCtor c;
#pragma acc parallel private(c)
  ;
}

// CHECK:      acc.private.recipe @privatization__ZTS14HasCoercedCtor : !cir.ptr<!rec_HasCoercedCtor> init {
// CHECK-NEXT:   ^bb0(%{{[^,)]+}}: !cir.ptr<!rec_HasCoercedCtor>
// CHECK-NEXT:   %[[COERCE:.*]] = cir.alloca "coerce" {{.*}} : !cir.ptr<!rec_Pair>
// CHECK-NEXT:   cir.alloca "openacc.private.init"
// CHECK:        cir.store %{{.+}}, %[[COERCE]] : !rec_Pair, !cir.ptr<!rec_Pair>
// CHECK:        %[[VIEW:.*]] = cir.cast bitcast %[[COERCE]] : !cir.ptr<!rec_Pair> -> !cir.ptr<!rec_anon_struct>
// CHECK:        %[[A:.*]] = cir.load %{{.+}} : !cir.ptr<!s64i>, !s64i
// CHECK:        %[[B:.*]] = cir.load %{{.+}} : !cir.ptr<!s64i>, !s64i
// CHECK:        cir.call @_ZN14HasCoercedCtorC1E4Pair(%{{.+}}, %[[A]], %[[B]]) : (!cir.ptr<!rec_HasCoercedCtor> {{.*}}, !s64i, !s64i) -> ()
