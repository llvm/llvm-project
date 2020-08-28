// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -debug-info-kind=limited %s -o - | FileCheck %s

// Run again with -gline-tables-only or -gline-directives-only and verify we don't crash.  We won't output
// type info at all.
// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -debug-info-kind=line-tables-only %s -o - | FileCheck %s -check-prefix LINES-ONLY
// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -debug-info-kind=line-directives-only %s -o - | FileCheck %s -check-prefix LINES-ONLY

// LINES-ONLY-NOT: !DICompositeType(tag: DW_TAG_structure_type

// "h" is at the top because it's in the compile unit's retainedTypes: list.
// CHECK: DICompositeType(tag: DW_TAG_structure_type, name: "h<int>"
// CHECK-NOT: DIFlagFwdDecl
// CHECK-SAME: ){{$}}

template <typename T>
struct a {
};
extern template class a<int>;
// CHECK-NOT: DICompositeType(tag: DW_TAG_structure_type, name: "a<int>"

template <typename T>
struct b {
};
extern template class b<int>;
b<int> bi;

template <typename T>
struct c {
  void f() {}
};
extern template class c<int>;
c<int> ci;
// CHECK: DICompositeType(tag: DW_TAG_structure_type, name: "c<int>"
// CHECK-SAME: DIFlagFwdDecl

template <typename T>
struct d {
  void f();
};
extern template class d<int>;
d<int> di;
// CHECK: DICompositeType(tag: DW_TAG_structure_type, name: "d<int>"
// CHECK-NOT: DIFlagFwdDecl
// CHECK-SAME: ){{$}}

template <typename T>
struct e {
  void f();
};
template <typename T>
void e<T>::f() {
}
extern template class e<int>;
e<int> ei;
// There's no guarantee that the out of line definition will appear before the
// explicit template instantiation definition, so conservatively emit the type
// definition here.
// CHECK: DICompositeType(tag: DW_TAG_structure_type, name: "e<int>"
// CHECK-NOT: DIFlagFwdDecl
// CHECK-SAME: ){{$}}

template <typename T>
struct f {
  void g();
};
extern template class f<int>;
template <typename T>
void f<T>::g() {
}
f<int> fi;
// CHECK: DICompositeType(tag: DW_TAG_structure_type, name: "f<int>"
// CHECK-NOT: DIFlagFwdDecl
// CHECK-SAME: ){{$}}

template <typename T>
struct g {
  void f();
};
template <>
void g<int>::f();
extern template class g<int>;
g<int> gi;
// CHECK: DICompositeType(tag: DW_TAG_structure_type, name: "g<int>"
// CHECK-NOT: DIFlagFwdDecl
// CHECK-SAME: ){{$}}

template <typename T>
struct h {
};
template class h<int>;

template <typename T>
struct i {
  void f() {}
};
template<> void i<int>::f();
extern template class i<int>;
i<int> ii;
// CHECK: DICompositeType(tag: DW_TAG_structure_type, name: "i<int>"
// CHECK-NOT: DIFlagFwdDecl
// CHECK-SAME: ){{$}}

template <typename T1, typename T2 = T1>
struct j {
};
extern template class j<int>;
j<int> jj;
template <typename T>
struct j_wrap {
};
j_wrap<j<int>> j_wrap_j;
// CHECK: DICompositeType(tag: DW_TAG_structure_type, name: "j<int, int>"
// CHECK: DICompositeType(tag: DW_TAG_structure_type, name: "j_wrap<j<int, int> >"

template <typename T>
struct k {
};
template <>
struct k<int>;
template struct k<int>;
// CHECK-NOT: !DICompositeType(tag: DW_TAG_structure_type, name: "k<int>"

// CHECK: DICompositeType(tag: DW_TAG_structure_type, name: "b<int>"
// CHECK-NOT: DIFlagFwdDecl
// CHECK-SAME: ){{$}}
