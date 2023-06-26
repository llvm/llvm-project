// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// No-proto definition followed by a correct call.
int noProto0(x) int x; { return x; }
int test0(int x) {
  // CHECK: cir.func @test0
  return noProto0(x); // We know the definition. Should be a direct call.
  // CHECK: %{{.+}} = cir.call @noProto0(%{{.+}})
}

// Declaration without prototype followed by its definition, then a correct call.
//
// Call to no-proto is made after definition, so a direct call can be used.
int noProto1();
int noProto1(int x) { return x; }
// CHECK: cir.func @noProto1(%arg0: !s32i {{.+}}) -> !s32i {
int test1(int x) {
  // CHECK: cir.func @test1
  return noProto1(x);
  // CHECK: %{{.+}} = cir.call @noProto1(%{{[0-9]+}}) : (!s32i) -> !s32i
}
