// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -mmlir --mlir-print-ir-before=cir-call-conv-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefix=BEFORE
// RUN: cir-opt %t.cir -o %t.once.cir
// RUN: cir-opt %t.once.cir --cir-call-conv-lowering=target=x86_64 -o %t.twice.cir
// RUN: diff %t.once.cir %t.twice.cir
// RUN: FileCheck %s --input-file=%t.once.cir --check-prefix=AFTER \
// RUN:   --implicit-check-not=requires_call_conv_lowering

// A lowered signature uses ordinary CIR types, so CallConvLowering cannot tell
// it apart from one that still needs lowering: struct V2 lowers to
// vector<2 x float>, which the pass would lower again to double, as it does
// for the source-level vector v2f. A function says it still needs lowering,
// the pass drops that once it is lowered, and a second run on the pass's own
// output finds nothing left to do.

struct V2 { float x, y; };
typedef float v2f __attribute__((vector_size(8)));

struct V2 mk(float a) { struct V2 v = {a, a}; return v; }
v2f vec(v2f a) { return a; }
struct V2 call_indirect(struct V2 (*fp)(float)) { return fp(1.0f); }

// BEFORE: cir.func {{.*}}@mk(%arg0: !cir.float {{.*}}) -> !rec_V2 requires_call_conv_lowering attributes
// BEFORE: cir.func {{.*}}@vec(%arg0: !cir.vector<2 x !cir.float> {{.*}}) -> !cir.vector<2 x !cir.float> requires_call_conv_lowering attributes
// BEFORE: cir.func {{.*}}@call_indirect(%arg0: !cir.ptr<!cir.func<(!cir.float) -> !rec_V2>> {{.*}}) -> !rec_V2 requires_call_conv_lowering attributes

// AFTER: cir.func {{.*}}@mk(%arg0: !cir.float {{.*}}) -> !cir.vector<2 x !cir.float> attributes
// AFTER: cir.func {{.*}}@vec(%arg0: !cir.double {{.*}}) -> !cir.double attributes
// AFTER: cir.func {{.*}}@call_indirect(%arg0: !cir.ptr<!cir.func<(!cir.float) -> !rec_V2>> {{.*}}) -> !cir.vector<2 x !cir.float> attributes
// AFTER:   cir.call %{{.*}}(%{{.*}}) : (!cir.ptr<!cir.func<(!cir.float) -> !cir.vector<2 x !cir.float>>>, !cir.float {{.*}}) -> !cir.vector<2 x !cir.float>
