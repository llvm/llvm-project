// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -emit-llvm \
// RUN:   -debug-info-kind=standalone -o - %s | FileCheck %s

using uint16_t = unsigned short;
using uint32_t = unsigned int;
using uint64_t = unsigned long long;

struct Test {
  static inline constexpr char STR[] = "Hello";
  static inline constexpr int NUMS[] = {1, 2, 3};
  static inline constexpr uint16_t SHORTS[] = {256, 512, 1024};
  static inline constexpr uint32_t INTS[] = {70000, 80000};
  static inline constexpr uint64_t LONGS[] = {4294967296ULL, 123456789ULL};
  static inline constexpr int NEG[] = {-1, -128, 42};
  // 128-bit integers and floating-point arrays should NOT produce extraData
  // (not yet supported).
  static inline constexpr __int128_t I128[] = {1, -1};
  static inline constexpr __uint128_t U128[] = {1, 2};
  static inline constexpr float FLOATS[] = {1.0f, 2.0f};
  static inline constexpr double DOUBLES[] = {1.0, 2.0};
};

void use(const void*);
void test() {
  use(&Test::STR);
  use(&Test::NUMS);
  use(&Test::SHORTS);
  use(&Test::INTS);
  use(&Test::LONGS);
  use(&Test::NEG);
  use(&Test::I128);
  use(&Test::U128);
  use(&Test::FLOATS);
  use(&Test::DOUBLES);
}

// Integer arrays: extraData on DIDerivedType member declarations.
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "STR",{{.*}}extraData: [6 x i8] c"Hello\00")
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "NUMS",{{.*}}extraData: [3 x i32] [i32 1, i32 2, i32 3])
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "SHORTS",{{.*}}extraData: [3 x i16] [i16 256, i16 512, i16 1024])
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "INTS",{{.*}}extraData: [2 x i32] [i32 70000, i32 80000])
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "LONGS",{{.*}}extraData: [2 x i64] [i64 4294967296, i64 123456789])
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "NEG",{{.*}}extraData: [3 x i32] [i32 -1, i32 -128, i32 42])
// 128-bit integers: no extraData (not yet supported).
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "I128",{{.*}}flags: DIFlagStaticMember)
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "U128",{{.*}}flags: DIFlagStaticMember)
// CHECK-NOT: name: "I128",{{.*}}extraData:
// CHECK-NOT: name: "U128",{{.*}}extraData:
// Floating-point arrays: no extraData.
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "FLOATS",{{.*}}flags: DIFlagStaticMember)
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "DOUBLES",{{.*}}flags: DIFlagStaticMember)
// CHECK-NOT: name: "FLOATS",{{.*}}extraData:
// CHECK-NOT: name: "DOUBLES",{{.*}}extraData:
