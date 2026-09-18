// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fclang-abi-compat=23 -emit-llvm -o - %s | FileCheck %s --check-prefix=V23

// GCC has a single entity per namespace and every reopening of an inline
// namespace adds its abi_tags to that entity. The expected manglings below are
// the ones produced by GCC. Clang <= 23 only used the tags from the first
// declaration of the namespace.

// The first declaration has no tag, a reopening adds one.
inline namespace AddedLater { struct S0 {}; }
inline namespace AddedLater __attribute__((abi_tag("X"))) { struct S1 {}; }
inline namespace AddedLater { struct S2 {}; }
S0 added0() { return {}; }
S1 added1() { return {}; }
S2 added2() { return {}; }
S1 added_var;
// CHECK-DAG: @_Z9added_varB1X =
// CHECK-DAG: define {{.*}} @_Z6added0B1Xv(
// CHECK-DAG: define {{.*}} @_Z6added1B1Xv(
// CHECK-DAG: define {{.*}} @_Z6added2B1Xv(
// V23-DAG: @added_var =
// V23-DAG: define {{.*}} @_Z6added0v(
// V23-DAG: define {{.*}} @_Z6added1v(
// V23-DAG: define {{.*}} @_Z6added2v(

// Reopenings with different tags: every tag applies.
inline namespace Different __attribute__((abi_tag("A"))) { struct S1 {}; }
inline namespace Different __attribute__((abi_tag("B"))) { struct S2 {}; }
inline namespace Different __attribute__((abi_tag("A"))) { struct S3 {}; }
Different::S1 diff1() { return {}; }
Different::S2 diff2() { return {}; }
Different::S3 diff3() { return {}; }
// CHECK-DAG: define {{.*}} @_Z5diff1B1AB1Bv(
// CHECK-DAG: define {{.*}} @_Z5diff2B1AB1Bv(
// CHECK-DAG: define {{.*}} @_Z5diff3B1AB1Bv(
// V23-DAG: define {{.*}} @_Z5diff1B1Av(
// V23-DAG: define {{.*}} @_Z5diff2B1Av(
// V23-DAG: define {{.*}} @_Z5diff3B1Av(

inline namespace Multi __attribute__((abi_tag("A", "B"))) { struct S1 {}; }
inline namespace Multi __attribute__((abi_tag("X", "Y", "B"))) { struct S2 {}; }
Multi::S1 multi1() { return {}; }
Multi::S2 multi2() { return {}; }
// CHECK-DAG: define {{.*}} @_Z6multi1B1AB1BB1XB1Yv(
// CHECK-DAG: define {{.*}} @_Z6multi2B1AB1BB1XB1Yv(
// V23-DAG: define {{.*}} @_Z6multi1B1AB1Bv(
// V23-DAG: define {{.*}} @_Z6multi2B1AB1Bv(

// Reopening without the attribute (the libstdc++ idiom) keeps the tag.
namespace std2 {
inline namespace __cxx11 __attribute__((__abi_tag__("cxx11"))) {}
}
namespace std2 {
namespace __cxx11 { template <class C> struct basic_string {}; }
typedef basic_string<char> string;
}
std2::string str() { return {}; }
std2::string str_var;
// CHECK-DAG: @_Z7str_varB5cxx11 =
// CHECK-DAG: define {{.*}} @_Z3strB5cxx11v(
// V23-DAG: @_Z7str_varB5cxx11 =
// V23-DAG: define {{.*}} @_Z3strB5cxx11v(

// Reopening with a subset of the tags.
inline namespace Subset __attribute__((abi_tag("A", "B"))) { struct S1 {}; }
inline namespace Subset __attribute__((abi_tag("A"))) { struct S2 {}; }
Subset::S1 subset1() { return {}; }
Subset::S2 subset2() { return {}; }
// CHECK-DAG: define {{.*}} @_Z7subset1B1AB1Bv(
// CHECK-DAG: define {{.*}} @_Z7subset2B1AB1Bv(
// V23-DAG: define {{.*}} @_Z7subset1B1AB1Bv(
// V23-DAG: define {{.*}} @_Z7subset2B1AB1Bv(

// Several attributes on one declaration.
inline namespace TwoAttrs __attribute__((abi_tag("A"))) __attribute__((abi_tag("B"))) { struct S {}; }
TwoAttrs::S two_attrs() { return {}; }
// CHECK-DAG: define {{.*}} @_Z9two_attrsB1AB1Bv(
// V23-DAG: define {{.*}} @_Z9two_attrsB1Av(

// Nested inline namespaces, both reopened with new tags.
inline namespace Outer __attribute__((abi_tag("O1"))) {
inline namespace Inner __attribute__((abi_tag("I1"))) { struct S {}; }
}
inline namespace Outer __attribute__((abi_tag("O2"))) {
inline namespace Inner __attribute__((abi_tag("I2"))) {}
}
Outer::Inner::S nested() { return {}; }
// CHECK-DAG: define {{.*}} @_Z6nestedB2I1B2I2B2O1B2O2v(
// V23-DAG: define {{.*}} @_Z6nestedB2I1B2O1v(

// Templates, static data members and explicitly tagged entities.
template <class T> Different::S1 tmpl(T) { return {}; }
template Different::S1 tmpl<int>(int);
template <class T> struct Holder { static Different::S2 member; };
template <class T> Different::S2 Holder<T>::member;
template struct Holder<int>;
__attribute__((abi_tag("Z"))) Different::S1 explicit_tag() { return {}; }
// CHECK-DAG: define {{.*}} @_Z4tmplIiEN9Different2S1ET_(
// CHECK-DAG: @_ZN6HolderIiE6memberB1AB1BE =
// CHECK-DAG: define {{.*}} @_Z12explicit_tagB1AB1BB1Zv(
// V23-DAG: define {{.*}} @_Z4tmplIiEN9Different2S1ET_(
// V23-DAG: @_ZN6HolderIiE6memberB1AE =
// V23-DAG: define {{.*}} @_Z12explicit_tagB1AB1Zv(

// abi_tag on a non-inline namespace is still ignored.
namespace NonInline {}
namespace NonInline __attribute__((abi_tag("X"))) { struct S {}; }
NonInline::S non_inline() { return {}; }
// CHECK-DAG: define {{.*}} @_Z10non_inlinev(
// V23-DAG: define {{.*}} @_Z10non_inlinev(
