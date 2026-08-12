// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:     -clangir-enable-call-conv-lowering -emit-cir %s -o %t.cir 2>&1 \
// RUN:   | FileCheck %s

struct S1 { short x, y, z; };

// The wrapper that inflates the value to the atomic size holds a pad member,
// and the classifier does not tell padding from data yet.
// CHECK: error: 'cir.func' op x86_64 calling-convention lowering not yet implemented for type '!cir.struct<padded {{.*}}pad !cir.array<!cir.int<s, 8> x 2>}>'
void take_atomic(_Atomic struct S1 s) {}
