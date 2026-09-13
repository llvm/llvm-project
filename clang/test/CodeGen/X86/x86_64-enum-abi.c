// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclang-abi-compat=23 -emit-llvm %s -o - | FileCheck %s

enum E { A, B };

struct EnumRecord {
  enum E value;
};

// CHECK-LABEL: define dso_local void @take_enum_record(
// CHECK-SAME: i32 %{{.*}})
void take_enum_record(struct EnumRecord value) {}

// CHECK-LABEL: define dso_local i32 @return_enum_record(
struct EnumRecord return_enum_record(struct EnumRecord value) { return value; }
