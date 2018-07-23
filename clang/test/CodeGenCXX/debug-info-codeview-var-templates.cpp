// RUN: %clang_cc1 %s -std=c++14 -triple=i686-pc-windows-msvc -debug-info-kind=limited -gcodeview -emit-llvm -o - | FileCheck %s

// Don't emit static data member debug info for variable templates.
// PR38004

struct TestImplicit {
  template <typename T>
  static const __SIZE_TYPE__ size_var = sizeof(T);
};
int instantiate_test1() { return TestImplicit::size_var<int> + TestImplicit::size_var<TestImplicit>; }
TestImplicit gv1;

// CHECK: ![[empty:[0-9]+]] = !{}

// CHECK: ![[A:[^ ]*]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "TestImplicit",
// CHECK-SAME: elements: ![[empty]]

template <typename T> bool vtpl;
struct TestSpecialization {
  template <typename T, typename U> static const auto sdm = vtpl<T>;
  template <> static const auto sdm<int, int> = false;
} gv2;

// CHECK: ![[A:[^ ]*]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "TestSpecialization",
// CHECK-SAME: elements: ![[empty]]

template <class> bool a;
template <typename> struct b;
struct TestPartial {
  template <typename... e> static auto d = a<e...>;
  template <typename... e> static auto d<b<e...>> = d<e...>;
} c;

// CHECK: ![[A:[^ ]*]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "TestPartial",
// CHECK-SAME: elements: ![[empty]]
