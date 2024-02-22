// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

//===----------------------------------------------------------------------===//
// DEFINED BEHAVIOUR
//===----------------------------------------------------------------------===//

// No-proto definition followed by a correct call.
int noProto0(x) int x; { return x; }
// CHECK: cir.func no_proto @noProto0(%arg0: !s32i {{.+}}) -> !s32i
int test0(int x) {
  // CHECK: cir.func @test0
  return noProto0(x); // We know the definition. Should be a direct call.
  // CHECK: %{{.+}} = cir.call @noProto0(%{{.+}})
}

// Declaration without prototype followed by its definition, then a correct call.
//
// Prototyped definition overrides no-proto declaration before any call is made,
// only allowing calls with proper arguments. This is the only case where the
// definition is not marked as no-proto.
int noProto1();
int noProto1(int x) { return x; }
// CHECK: cir.func @noProto1(%arg0: !s32i {{.+}}) -> !s32i
int test1(int x) {
  // CHECK: cir.func @test1
  return noProto1(x);
  // CHECK: %{{.+}} = cir.call @noProto1(%{{[0-9]+}}) : (!s32i) -> !s32i
}

// Declaration without prototype followed by a correct call, then its definition.
//
// Call to no-proto is made before definition, so a variadic call that takes anything
// is created. Later, when the definition is found, no-proto is replaced.
int noProto2();
int test2(int x) {
  return noProto2(x);
  // CHECK:  [[GGO:%.*]] = cir.get_global @noProto2 : cir.ptr <!cir.func<!s32i (!s32i)>>
  // CHECK:  [[CAST:%.*]] = cir.cast(bitcast, %3 : !cir.ptr<!cir.func<!s32i (!s32i)>>), !cir.ptr<!cir.func<!s32i (!s32i)>>
  // CHECK:  {{.*}} = cir.call [[CAST]](%{{[0-9]+}}) : (!cir.ptr<!cir.func<!s32i (!s32i)>>, !s32i) -> !s32i
}
int noProto2(int x) { return x; }
// CHECK: cir.func no_proto @noProto2(%arg0: !s32i {{.+}}) -> !s32i

// No-proto declaration without definition (any call here is "correct").
//
// Call to no-proto is made before definition, so a variadic call that takes anything
// is created. Definition is not in the translation unit, so it is left as is.
int noProto3();
// cir.func private no_proto @noProto3(...) -> !s32i
int test3(int x) {
// CHECK: cir.func @test3
  return noProto3(x);
  // CHECK:  [[GGO:%.*]] = cir.get_global @noProto3 : cir.ptr <!cir.func<!s32i (...)>>
  // CHECK:  [[CAST:%.*]] = cir.cast(bitcast, [[GGO]] : !cir.ptr<!cir.func<!s32i (...)>>), !cir.ptr<!cir.func<!s32i (!s32i)>>
  // CHECK:  {{%.*}} = cir.call [[CAST]](%{{[0-9]+}}) : (!cir.ptr<!cir.func<!s32i (!s32i)>>, !s32i) -> !s32i
}


//===----------------------------------------------------------------------===//
// UNDEFINED BEHAVIOUR
//
// No-proto definitions followed by incorrect calls.
//===----------------------------------------------------------------------===//

// No-proto definition followed by an incorrect call due to extra args.
int noProto4() { return 0; }
// cir.func private no_proto @noProto4() -> !s32i
int test4(int x) {
  return noProto4(x); // Even if we know the definition, this should compile.
  // CHECK:  [[GGO:%.*]] = cir.get_global @noProto4 : cir.ptr <!cir.func<!s32i ()>>
  // CHECK:  [[CAST:%.*]] = cir.cast(bitcast, [[GGO]] : !cir.ptr<!cir.func<!s32i ()>>), !cir.ptr<!cir.func<!s32i (!s32i)>>
  // CHECK:  {{%.*}} = cir.call [[CAST]]({{%.*}}) : (!cir.ptr<!cir.func<!s32i (!s32i)>>, !s32i) -> !s32i
}

// No-proto definition followed by an incorrect call due to lack of args.
int noProto5();
int test5(int x) {
  return noProto5();
  // CHECK:  [[GGO:%.*]] = cir.get_global @noProto5 : cir.ptr <!cir.func<!s32i (!s32i)>>
  // CHECK:  [[CAST:%.*]] = cir.cast(bitcast, [[GGO]] : !cir.ptr<!cir.func<!s32i (!s32i)>>), !cir.ptr<!cir.func<!s32i ()>>
  // CHECK:  {{%.*}} = cir.call [[CAST]]() : (!cir.ptr<!cir.func<!s32i ()>>) -> !s32i
}
int noProto5(int x) { return x; }
// CHECK: cir.func no_proto @noProto5(%arg0: !s32i {{.+}}) -> !s32i
