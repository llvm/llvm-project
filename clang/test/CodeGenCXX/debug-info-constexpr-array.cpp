// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -emit-llvm \
// RUN:   -debug-info-kind=standalone -o - %s | FileCheck %s

struct Test {
  static inline constexpr char STR[] = "Hello";
  static inline constexpr int NUMS[] = {1, 2, 3};
  static inline constexpr unsigned char BYTES[] = {0xDE, 0xAD};
};

void use() {
  (void)Test::STR;
  (void)Test::NUMS;
  (void)Test::BYTES;
}

// "Hello\0" as [6 x i8]
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "STR"{{.*}}extraData: [6 x i8] c"Hello\00"

// {1, 2, 3} as [3 x i32]
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "NUMS"{{.*}}extraData: [3 x i32] [i32 1, i32 2, i32 3]

// {0xDE, 0xAD} as [2 x i8]
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "BYTES"{{.*}}extraData: [2 x i8] c"\DE\AD"