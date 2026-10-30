// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: cir-opt %t.cir -o %t.once.cir
// RUN: cir-opt %t.once.cir --cir-call-conv-lowering=target=x86_64 -o %t.twice.cir
// RUN: diff %t.once.cir %t.twice.cir
// RUN: FileCheck %s --input-file=%t.once.cir

// A lowered signature uses ordinary CIR types, so CallConvLowering cannot tell
// it apart from one that still needs lowering: struct V2 lowers to
// vector<2 x float>, which the pass would lower again to double, as it does
// for the source-level vector v2f. The pass marks what it has lowered and
// skips it when run again on its own output.

struct V2 { float x, y; };
typedef float v2f __attribute__((vector_size(8)));

struct V2 mk(float a) { struct V2 v = {a, a}; return v; }
v2f vec(v2f a) { return a; }
struct V2 call_indirect(struct V2 (*fp)(float)) { return fp(1.0f); }

// CHECK: cir.func {{.*}}@mk(%arg0: !cir.float {{.*}}) -> !cir.vector<2 x !cir.float> call_conv_lowered attributes
// CHECK: cir.func {{.*}}@vec(%arg0: !cir.double {{.*}}) -> !cir.double call_conv_lowered attributes
// CHECK: cir.func {{.*}}@call_indirect(%arg0: !cir.ptr<!cir.func<(!cir.float) -> !rec_V2>> {{.*}}) -> !cir.vector<2 x !cir.float> call_conv_lowered attributes
// CHECK:   cir.call %{{.*}}(%{{.*}}) : (!cir.ptr<!cir.func<(!cir.float) -> !cir.vector<2 x !cir.float>>>, !cir.float {{.*}}) -> !cir.vector<2 x !cir.float> call_conv_lowered
