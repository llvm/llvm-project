// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

void array() {
  float arr[4];

  auto &[first, second, third, fourth] = arr;

  [=]() { return first;}();
  // CIR-LABEL: cir.func {{.*}}@_ZZ5arrayvENK3$_0clEv
  // CIR: %[[THIS:.*]] = cir.alloca !cir.ptr<![[LAMBDA_TY:.*]]>, !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, ["this", init]
  // CIR: %[[LOAD_THIS:.*]] = cir.load %[[THIS]] : !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, !cir.ptr<![[LAMBDA_TY]]>
  // CIR: %[[FIRST_MEM:.*]] = cir.get_member %[[LOAD_THIS]][0] {name = "first"} : !cir.ptr<![[LAMBDA_TY]]> -> !cir.ptr<!cir.float>
  // CIR: %[[GET_FIRST:.*]] = cir.load {{.*}}%[[FIRST_MEM]] : !cir.ptr<!cir.float>, !cir.float
  //
  // LLVM-LABEL: define {{.*}}@"_ZZ5arrayvENK3$_0clEv"
  // LLVM:   %[[THIS:.*]] = alloca ptr
  // LLVM:   %[[LOAD_THIS:.*]] = load ptr, ptr %[[THIS]]
  // LLVM:   %[[FIRST_MEM:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %[[LOAD_THIS]], i32 0, i32 0
  // LLVM:   %[[GET_FIRST:.*]] = load float, ptr %[[FIRST_MEM]]

  [&]() { return second;}();
  // CIR-LABEL: cir.func {{.*}}@_ZZ5arrayvENK3$_1clEv
  // CIR: %[[THIS:.*]] = cir.alloca !cir.ptr<![[LAMBDA_TY:.*]]>, !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, ["this", init]
  // CIR: %[[LOAD_THIS:.*]] = cir.load %[[THIS]] : !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, !cir.ptr<![[LAMBDA_TY]]>
  // CIR: %[[SECOND_MEM:.*]] = cir.get_member %[[LOAD_THIS]][0] {name = "second"} : !cir.ptr<![[LAMBDA_TY]]> -> !cir.ptr<!cir.ptr<!cir.float>>
  // CIR: %[[GET_SECOND:.*]] = cir.load {{.*}}%[[SECOND_MEM]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
  // CIR: %[[DEREF:.*]] = cir.load{{.*}} %[[GET_SECOND]] : !cir.ptr<!cir.float>, !cir.float
  //
  // LLVM-LABEL: define {{.*}}@"_ZZ5arrayvENK3$_1clEv"
  // LLVM:   %[[THIS:.*]] = alloca ptr
  // LLVM:   %[[LOAD_THIS:.*]] = load ptr, ptr %[[THIS]]
  // LLVM:   %[[SECOND_MEM:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %[[LOAD_THIS]], i32 0, i32 0
  // LLVM:   %[[GET_SECOND:.*]] = load ptr, ptr %[[SECOND_MEM]]
  // LLVM:   %[[DEREF:.*]] = load float, ptr %[[GET_SECOND]]

  [third]() { return third;}();
  // CIR-LABEL: cir.func {{.*}}@_ZZ5arrayvENK3$_2clEv
  // CIR: %[[THIS:.*]] = cir.alloca !cir.ptr<![[LAMBDA_TY:.*]]>, !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, ["this", init]
  // CIR: %[[LOAD_THIS:.*]] = cir.load %[[THIS]] : !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, !cir.ptr<![[LAMBDA_TY]]>
  // CIR: %[[THIRD_MEM:.*]] = cir.get_member %[[LOAD_THIS]][0] {name = "third"} : !cir.ptr<![[LAMBDA_TY]]> -> !cir.ptr<!cir.float>
  // CIR: %[[GET_THIRD:.*]] = cir.load {{.*}}%[[THIRD_MEM]] : !cir.ptr<!cir.float>, !cir.float
  //
  // LLVM-LABEL: define {{.*}}@"_ZZ5arrayvENK3$_2clEv"
  // LLVM:   %[[THIS:.*]] = alloca ptr
  // LLVM:   %[[LOAD_THIS:.*]] = load ptr, ptr %[[THIS]]
  // LLVM:   %[[THIRD_MEM:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %[[LOAD_THIS]], i32 0, i32 0
  // LLVM:   %[[GET_THIRD:.*]] = load float, ptr %[[THIRD_MEM]]

  [&fourth]() { return fourth;}();
  // CIR-LABEL: cir.func {{.*}}@_ZZ5arrayvENK3$_3clEv
  // CIR: %[[THIS:.*]] = cir.alloca !cir.ptr<![[LAMBDA_TY:.*]]>, !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, ["this", init]
  // CIR: %[[LOAD_THIS:.*]] = cir.load %[[THIS]] : !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, !cir.ptr<![[LAMBDA_TY]]>
  // CIR: %[[FOURTH_MEM:.*]] = cir.get_member %[[LOAD_THIS]][0] {name = "fourth"} : !cir.ptr<![[LAMBDA_TY]]> -> !cir.ptr<!cir.ptr<!cir.float>>
  // CIR: %[[GET_FOURTH:.*]] = cir.load {{.*}}%[[FOURTH_MEM]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
  // CIR: %[[DEREF:.*]] = cir.load{{.*}} %[[GET_FOURTH]] : !cir.ptr<!cir.float>, !cir.float
  //
  // LLVM-LABEL: define {{.*}}@"_ZZ5arrayvENK3$_3clEv"
  // LLVM:   %[[THIS:.*]] = alloca ptr
  // LLVM:   %[[LOAD_THIS:.*]] = load ptr, ptr %[[THIS]]
  // LLVM:   %[[FOURTH_MEM:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %[[LOAD_THIS]], i32 0, i32 0
  // LLVM:   %[[GET_FOURTH:.*]] = load ptr, ptr %[[FOURTH_MEM]]
  // LLVM:   %[[DEREF:.*]] = load float, ptr %[[GET_FOURTH]]
}

struct S { int a, b, c, d; };

void Struct() {
  S s;

  auto [first, second, third, fourth] = s;

  [=]() { return first;}();
  // CIR-LABEL: cir.func {{.*}}@_ZZ6StructvENK3$_0clEv
  // CIR: %[[THIS:.*]] = cir.alloca !cir.ptr<![[LAMBDA_TY:.*]]>, !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, ["this", init]
  // CIR: %[[LOAD_THIS:.*]] = cir.load %[[THIS]] : !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, !cir.ptr<![[LAMBDA_TY]]>
  // CIR: %[[FIRST_MEM:.*]] = cir.get_member %[[LOAD_THIS]][0] {name = "first"} : !cir.ptr<![[LAMBDA_TY]]> -> !cir.ptr<!s32i>
  // CIR: %[[LOAD_FIRST:.*]] = cir.load {{.*}}%[[FIRST_MEM]] : !cir.ptr<!s32i>, !s32i
  //
  // LLVM-LABEL: define {{.*}}@"_ZZ6StructvENK3$_0clEv"
  // LLVM:   %[[THIS:.*]] = alloca ptr
  // LLVM:   %[[LOAD_THIS:.*]] = load ptr, ptr %[[THIS]]
  // LLVM:   %[[FIRST_MEM:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %[[LOAD_THIS]], i32 0, i32 0
  // LLVM:   %[[GET_FIRST:.*]] = load i32, ptr %[[FIRST_MEM]]
  [&]() { return second;}();
  // CIR-LABEL: cir.func {{.*}}@_ZZ6StructvENK3$_1clEv
  // CIR: %[[THIS:.*]] = cir.alloca !cir.ptr<![[LAMBDA_TY:.*]]>, !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, ["this", init]
  // CIR: %[[LOAD_THIS:.*]] = cir.load %[[THIS]] : !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, !cir.ptr<![[LAMBDA_TY]]>
  // CIR: %[[SECOND_MEM:.*]] = cir.get_member %[[LOAD_THIS]][0] {name = "second"} : !cir.ptr<![[LAMBDA_TY]]> -> !cir.ptr<!cir.ptr<!s32i>>
  // CIR: %[[GET_SECOND:.*]] = cir.load {{.*}}%[[SECOND_MEM]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: %[[DEREF:.*]] = cir.load{{.*}} %[[GET_SECOND]] : !cir.ptr<!s32i>, !s32i
  //
  // LLVM-LABEL: define {{.*}}@"_ZZ6StructvENK3$_1clEv"
  // LLVM:   %[[THIS:.*]] = alloca ptr
  // LLVM:   %[[LOAD_THIS:.*]] = load ptr, ptr %[[THIS]]
  // LLVM:   %[[SECOND_MEM:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %[[LOAD_THIS]], i32 0, i32 0
  // LLVM:   %[[GET_SECOND:.*]] = load ptr, ptr %[[SECOND_MEM]]
  // LLVM:   %[[DEREF:.*]] = load i32, ptr %[[GET_SECOND]]
  [third]() { return third;}();
  // CIR-LABEL: cir.func {{.*}}@_ZZ6StructvENK3$_2clEv
  // CIR: %[[THIS:.*]] = cir.alloca !cir.ptr<![[LAMBDA_TY:.*]]>, !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, ["this", init]
  // CIR: %[[LOAD_THIS:.*]] = cir.load %[[THIS]] : !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, !cir.ptr<![[LAMBDA_TY]]>
  // CIR: %[[THIRD_MEM:.*]] = cir.get_member %[[LOAD_THIS]][0] {name = "third"} : !cir.ptr<![[LAMBDA_TY]]> -> !cir.ptr<!s32i>
  // CIR: %[[LOAD_THIRD:.*]] = cir.load {{.*}}%[[THIRD_MEM]] : !cir.ptr<!s32i>, !s32i
  //
  // LLVM-LABEL: define {{.*}}@"_ZZ6StructvENK3$_2clEv"
  // LLVM:   %[[THIS:.*]] = alloca ptr
  // LLVM:   %[[LOAD_THIS:.*]] = load ptr, ptr %[[THIS]]
  // LLVM:   %[[THIRD_MEM:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %[[LOAD_THIS]], i32 0, i32 0
  // LLVM:   %[[GET_THIRD:.*]] = load i32, ptr %[[THIRD_MEM]]
  [&fourth]() { return fourth;}();
  // CIR-LABEL: cir.func {{.*}}@_ZZ6StructvENK3$_3clEv
  // CIR: %[[THIS:.*]] = cir.alloca !cir.ptr<![[LAMBDA_TY:.*]]>, !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, ["this", init]
  // CIR: %[[LOAD_THIS:.*]] = cir.load %[[THIS]] : !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, !cir.ptr<![[LAMBDA_TY]]>
  // CIR: %[[FOURTH_MEM:.*]] = cir.get_member %[[LOAD_THIS]][0] {name = "fourth"} : !cir.ptr<![[LAMBDA_TY]]> -> !cir.ptr<!cir.ptr<!s32i>>
  // CIR: %[[GET_FOURTH:.*]] = cir.load {{.*}}%[[FOURTH_MEM]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: %[[DEREF:.*]] = cir.load{{.*}} %[[GET_FOURTH]] : !cir.ptr<!s32i>, !s32i
  //
  // LLVM-LABEL: define {{.*}}@"_ZZ6StructvENK3$_3clEv"
  // LLVM:   %[[THIS:.*]] = alloca ptr
  // LLVM:   %[[LOAD_THIS:.*]] = load ptr, ptr %[[THIS]]
  // LLVM:   %[[FOURTH_MEM:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %[[LOAD_THIS]], i32 0, i32 0
  // LLVM:   %[[GET_FOURTH:.*]] = load ptr, ptr %[[FOURTH_MEM]]
  // LLVM:   %[[DEREF:.*]] = load i32, ptr %[[GET_FOURTH]]
}

void StructNested() {
  S s;

  auto [first, second, third, fourth] = s;
  auto outer = [&]() {
  [=]() { return first;}();
  // CIR-LABEL: cir.func {{.*}}@_ZZZ12StructNestedvENK3$_0clEvENKUlvE_clEv(
  // CIR: %[[THIS:.*]] = cir.alloca !cir.ptr<![[LAMBDA_TY:.*]]>, !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, ["this", init]
  // CIR: %[[LOAD_THIS:.*]] = cir.load %[[THIS]] : !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, !cir.ptr<![[LAMBDA_TY]]>
  // CIR: %[[FIRST_MEM:.*]] = cir.get_member %[[LOAD_THIS]][0] {name = "first"} : !cir.ptr<![[LAMBDA_TY]]> -> !cir.ptr<!s32i>
  // CIR: %[[LOAD_FIRST:.*]] = cir.load {{.*}}%[[FIRST_MEM]] : !cir.ptr<!s32i>, !s32i
  //
  // LLVM-LABEL: define {{.*}}@"_ZZZ12StructNestedvENK3$_0clEvENKUlvE_clEv"
  // LLVM:   %[[THIS:.*]] = alloca ptr
  // LLVM:   %[[LOAD_THIS:.*]] = load ptr, ptr %[[THIS]]
  // LLVM:   %[[FIRST_MEM:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %[[LOAD_THIS]], i32 0, i32 0
  // LLVM:   %[[GET_FIRST:.*]] = load i32, ptr %[[FIRST_MEM]]
  [&]() { return second;}();
  // CIR-LABEL: cir.func {{.*}}@_ZZZ12StructNestedvENK3$_0clEvENKUlvE0_clEv(
  // CIR: %[[THIS:.*]] = cir.alloca !cir.ptr<![[LAMBDA_TY:.*]]>, !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, ["this", init]
  // CIR: %[[LOAD_THIS:.*]] = cir.load %[[THIS]] : !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, !cir.ptr<![[LAMBDA_TY]]>
  // CIR: %[[SECOND_MEM:.*]] = cir.get_member %[[LOAD_THIS]][0] {name = "second"} : !cir.ptr<![[LAMBDA_TY]]> -> !cir.ptr<!cir.ptr<!s32i>>
  // CIR: %[[GET_SECOND:.*]] = cir.load {{.*}}%[[SECOND_MEM]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: %[[DEREF:.*]] = cir.load{{.*}} %[[GET_SECOND]] : !cir.ptr<!s32i>, !s32i
  //
  // LLVM-LABEL: define {{.*}}@"_ZZZ12StructNestedvENK3$_0clEvENKUlvE0_clEv"
  // LLVM:   %[[THIS:.*]] = alloca ptr
  // LLVM:   %[[LOAD_THIS:.*]] = load ptr, ptr %[[THIS]]
  // LLVM:   %[[SECOND_MEM:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %[[LOAD_THIS]], i32 0, i32 0
  // LLVM:   %[[GET_SECOND:.*]] = load ptr, ptr %[[SECOND_MEM]]
  // LLVM:   %[[DEREF:.*]] = load i32, ptr %[[GET_SECOND]]
  [third]() { return third;}();
  // CIR-LABEL: cir.func {{.*}}@_ZZZ12StructNestedvENK3$_0clEvENKUlvE1_clEv(
  // CIR: %[[THIS:.*]] = cir.alloca !cir.ptr<![[LAMBDA_TY:.*]]>, !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, ["this", init]
  // CIR: %[[LOAD_THIS:.*]] = cir.load %[[THIS]] : !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, !cir.ptr<![[LAMBDA_TY]]>
  // CIR: %[[THIRD_MEM:.*]] = cir.get_member %[[LOAD_THIS]][0] {name = "third"} : !cir.ptr<![[LAMBDA_TY]]> -> !cir.ptr<!s32i>
  // CIR: %[[LOAD_THIRD:.*]] = cir.load {{.*}}%[[THIRD_MEM]] : !cir.ptr<!s32i>, !s32i
  //
  // LLVM-LABEL: define {{.*}}@"_ZZZ12StructNestedvENK3$_0clEvENKUlvE1_clEv"
  // LLVM:   %[[THIS:.*]] = alloca ptr
  // LLVM:   %[[LOAD_THIS:.*]] = load ptr, ptr %[[THIS]]
  // LLVM:   %[[THIRD_MEM:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %[[LOAD_THIS]], i32 0, i32 0
  // LLVM:   %[[GET_THIRD:.*]] = load i32, ptr %[[THIRD_MEM]]
  [&fourth]() { return fourth;}();
  // CIR-LABEL: cir.func {{.*}}@_ZZZ12StructNestedvENK3$_0clEvENKUlvE2_clEv(
  // CIR: %[[THIS:.*]] = cir.alloca !cir.ptr<![[LAMBDA_TY:.*]]>, !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, ["this", init]
  // CIR: %[[LOAD_THIS:.*]] = cir.load %[[THIS]] : !cir.ptr<!cir.ptr<![[LAMBDA_TY]]>>, !cir.ptr<![[LAMBDA_TY]]>
  // CIR: %[[FOURTH_MEM:.*]] = cir.get_member %[[LOAD_THIS]][0] {name = "fourth"} : !cir.ptr<![[LAMBDA_TY]]> -> !cir.ptr<!cir.ptr<!s32i>>
  // CIR: %[[GET_FOURTH:.*]] = cir.load {{.*}}%[[FOURTH_MEM]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: %[[DEREF:.*]] = cir.load{{.*}} %[[GET_FOURTH]] : !cir.ptr<!s32i>, !s32i
  //
  // LLVM-LABEL: define {{.*}}@"_ZZZ12StructNestedvENK3$_0clEvENKUlvE2_clEv"
  // LLVM:   %[[THIS:.*]] = alloca ptr
  // LLVM:   %[[LOAD_THIS:.*]] = load ptr, ptr %[[THIS]]
  // LLVM:   %[[FOURTH_MEM:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %[[LOAD_THIS]], i32 0, i32 0
  // LLVM:   %[[GET_FOURTH:.*]] = load ptr, ptr %[[FOURTH_MEM]]
  // LLVM:   %[[DEREF:.*]] = load i32, ptr %[[GET_FOURTH]]
  };

  outer();
}
