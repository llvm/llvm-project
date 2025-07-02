// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -debug-info-kind=standalone  -o - %s | FileCheck %s

struct X {
  typedef int inside;
  inside i;
};

template <typename T = int>
struct Y {
  typedef int outside;
  outside o;
};

X x;
Y<> y;

// CHECK: ![[Y:.*]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Y<int>", {{.*}}identifier: "_ZTS1YIiE")
// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "outside", scope: ![[Y]],
