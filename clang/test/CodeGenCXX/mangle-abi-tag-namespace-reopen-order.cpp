// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fcxx-exceptions \
// RUN:   -fexceptions -emit-llvm -o %t.ll %s
// RUN: FileCheck %s --input-file=%t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=EH
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=NEG
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fcxx-exceptions \
// RUN:   -fexceptions -fclang-abi-compat=23 -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefix=V23

// The abi_tags of an inline namespace can grow when the namespace is reopened.
// As in GCC, the tags that a function or variable gets from inline namespaces
// are fixed the first time it is mangled. Every name that embeds its encoding
// keeps seeing the namespaces as they were at that point, even if it is only
// mangled after a namespace got more tags. Otherwise the local class below
// would get two different type_info objects and the exception would not be
// caught.

// None of the names derived from an entity that was mangled before the
// reopening may pick up the tag that was added later. (FileCheck does not
// apply --implicit-check-not inside a group of CHECK-DAG lines, hence the
// separate run.)
// NEG-NOT: earlyB1AB1B
// NEG-NOT: early_emptyB1X
// NEG-NOT: mB1AB1B
// NEG-NOT: tlB1AB1B
// NEG-NOT: inline_funcB1AB1B
// NEG-NOT: inline_lambdaB1AB1B

template <class T> void thrower() { throw T(); }

inline namespace N __attribute__((abi_tag("A"))) { struct S {}; }

// Emitted before the tag "B" is added.
S early() {
  struct L {};
  try {
    // Instantiated at the end of the translation unit, after "B" was added.
    thrower<L>();
  } catch (L &) {
  }
  return {};
}

inline namespace N __attribute__((abi_tag("B"))) {}

// Entities that are first mangled after the reopening get all the tags.
S late() { return {}; }

// CHECK-DAG: @_ZTIZ5earlyB1AvE1L =
// CHECK-DAG: @_ZTSZ5earlyB1AvE1L =
// CHECK-DAG: define {{.*}} @_Z5earlyB1Av(
// CHECK-DAG: define {{.*}} @_Z7throwerIZ5earlyB1AvE1LEvv(
// CHECK-DAG: define {{.*}} @_Z4lateB1AB1Bv(
// V23-DAG: define {{.*}} @_Z5earlyB1Av(
// V23-DAG: define {{.*}} @_Z7throwerIZ5earlyB1AvE1LEvv(
// V23-DAG: define {{.*}} @_Z4lateB1Av(

// The handler and the throw expression have to use the same type_info.
// EH-LABEL: define {{.*}} @_Z5earlyB1Av(
// EH: catch ptr @_ZTIZ5earlyB1AvE1L
// EH-LABEL: define {{.*}} @_Z7throwerIZ5earlyB1AvE1LEvv(
// EH: call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIZ5earlyB1AvE1L, ptr null)

// A function can have no implicit tags at all when it is first mangled: the
// namespace only gets its first tag afterwards.
inline namespace Empty { struct SE {}; }
SE early_empty() {
  struct L {};
  try {
    thrower<L>();
  } catch (L &) {
  }
  return {};
}
inline namespace Empty __attribute__((abi_tag("X"))) {}
SE late_empty() { return {}; }

// CHECK-DAG: @_ZTIZ11early_emptyvE1L =
// CHECK-DAG: define {{.*}} @_Z11early_emptyv(
// CHECK-DAG: define {{.*}} @_Z7throwerIZ11early_emptyvE1LEvv(
// CHECK-DAG: define {{.*}} @_Z10late_emptyB1Xv(
// V23-DAG: define {{.*}} @_Z11early_emptyv(
// V23-DAG: define {{.*}} @_Z10late_emptyv(
// EH-LABEL: define {{.*}} @_Z11early_emptyv(
// EH: catch ptr @_ZTIZ11early_emptyvE1L
// EH-LABEL: define {{.*}} @_Z7throwerIZ11early_emptyvE1LEvv(
// EH: call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIZ11early_emptyvE1L, ptr null)

// Variables: the name of the variable is mangled at its first use, the guard
// variable and the thread_local helpers when the definition is instantiated at
// the end of the translation unit.
inline namespace NV __attribute__((abi_tag("A"))) { struct SV { SV(); }; }
template <class T> struct Holder { static SV m; };
template <class T> SV Holder<T>::m;
SV *use_m() { return &Holder<int>::m; }
template <class T> thread_local SV tl;
SV *use_tl() { return &tl<int>; }
inline namespace NV __attribute__((abi_tag("B"))) {}

// CHECK-DAG: @_ZN6HolderIiE1mB1AE =
// CHECK-DAG: @_ZGVN6HolderIiE1mB1AE =
// CHECK-DAG: @_Z2tlB1AIiE =
// CHECK-DAG: @_ZGV2tlB1AIiE =
// CHECK-DAG: @_ZTH2tlB1AIiE =
// CHECK-DAG: define {{.*}} @_ZTW2tlB1AIiE(
// V23-DAG: @_ZN6HolderIiE1mB1AE =
// V23-DAG: @_ZGVN6HolderIiE1mB1AE =

// An inline function is mangled at its definition but its body is only emitted
// at the end of the translation unit.
inline namespace NI __attribute__((abi_tag("A"))) { struct SI { SI(); }; }
inline SI inline_func() {
  static int counter;
  ++counter;
  return {};
}
inline SI inline_lambda() {
  auto l = [] { return 1; };
  l();
  return {};
}
void use_inline() { inline_func(); inline_lambda(); }
inline namespace NI __attribute__((abi_tag("B"))) {}

// CHECK-DAG: @_ZZ11inline_funcB1AvE7counter =
// CHECK-DAG: define {{.*}} @_Z11inline_funcB1Av(
// CHECK-DAG: define {{.*}} @_Z13inline_lambdaB1Av(
// CHECK-DAG: define {{.*}} @_ZZ13inline_lambdaB1AvENKUlvE_clEv(
// V23-DAG: @_ZZ11inline_funcB1AvE7counter =
// V23-DAG: define {{.*}} @_ZZ13inline_lambdaB1AvENKUlvE_clEv(
