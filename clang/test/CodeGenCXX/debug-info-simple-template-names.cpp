// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown_unknown -debug-info-kind=limited -gsimple-template-names=mangled %s -o - -w -std=c++17 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown_unknown -debug-info-kind=limited -gsimple-template-names=simple %s -o - -w -std=c++17 | FileCheck --check-prefix=SIMPLE --implicit-check-not=_STN %s
// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown_unknown -debug-info-kind=limited %s -o - -w -std=c++17 | FileCheck --check-prefix=FULL --implicit-check-not=_STN %s

template <typename... T>
void f1() {}
template <typename T, T V>
void f2() {}
template <typename... T>
struct t1 {};
extern int x;
int x;
struct t2 {
  template <typename T = float>
  operator t1<int>() { __builtin_unreachable(); }
};
template <template <typename...> class T>
void f3() {}
namespace {
enum LocalEnum { LocalEnum1 };
}
template<typename T, T ... ts>
struct t3 { };
struct t4 {
  t3<LocalEnum, LocalEnum1> m1;
};
  
t4 v1;
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "t3<(anonymous namespace)::LocalEnum, (anonymous namespace)::LocalEnum1>"
void f() {
  // Basic examples of simplifiable/rebuildable names
  f1<>();
  // CHECK: !DISubprogram(name: "_STNf1|<>",
  // SIMPLE: !DISubprogram(name: "f1",
  // FULL: !DISubprogram(name: "f1<>",
  f1<int>();
  // CHECK: !DISubprogram(name: "_STNf1|<int>",
  f1<void()>();
  // CHECK: !DISubprogram(name: "_STNf1|<void ()>",
  f2<int, 42>();
  // CHECK: !DISubprogram(name: "_STNf2|<int, 42>",

  // Check that even though the nested name can't be rebuilt, it'll carry its
  // full name and the outer name can be rebuilt from that.
  f1<t1<void() noexcept>>();
  // CHECK: !DISubprogram(name: "_STNf1|<t1<void () noexcept> >",

  // Vector array types are encoded in DWARF but the decoding in llvm-dwarfdump
  // isn't implemented yet.
  f1<__attribute__((__vector_size__((sizeof(int) * 2)))) int>();
  // CHECK: !DISubprogram(name: "f1<__attribute__((__vector_size__(2 * sizeof(int)))) int>",

  // noexcept is part of function types in C++17 onwards, but not encoded in
  // DWARF
  f1<void() noexcept>();
  // CHECK: !DISubprogram(name: "f1<void () noexcept>",

  // Unnamed entities (lambdas, structs/classes, enums) can't be fully rebuilt
  // since we don't emit the column number. Also lambdas and unnamed classes are
  // ambiguous with each other - there's no DWARF that designates a lambda as
  // anything other than another unnamed class/struct.
  auto Lambda = [] {};
  f1<decltype(Lambda)>();
  // CHECK: !DISubprogram(name: "f1<(lambda at {{.*}}debug-info-simple-template-names.cpp:[[# @LINE - 2]]:17)>",
  f1<t1<t1<decltype(Lambda)>>>();
  // CHECK: !DISubprogram(name: "f1<t1<t1<(lambda at {{.*}}> > >",
  struct {
  } unnamed_struct;
  f1<decltype(unnamed_struct)>();
  // CHECK: !DISubprogram(name: "f1<(unnamed struct at {{.*}}debug-info-simple-template-names.cpp:[[# @LINE - 3]]:3)>",
  f1<void (decltype(unnamed_struct))>();
  // CHECK: !DISubprogram(name: "f1<void ((unnamed struct at {{.*}}debug-info-simple-template-names.cpp:[[# @LINE - 5]]:3))>",
  enum {} unnamed_enum;
  f1<decltype(unnamed_enum)>();
  // CHECK: !DISubprogram(name: "f1<(unnamed enum at {{.*}}debug-info-simple-template-names.cpp:[[# @LINE - 2]]:3)>",

  // Declarations can't readily be reversed as the value in the DWARF only
  // contains the address of the value - we'd have to do symbol lookup to find
  // the name of that value (& rely on it not having been stripped out, etc).
  f2<int *, &x>();
  // CHECK: !DISubprogram(name: "f2<int *, &x>",

  // We could probably handle \/ this case, but since it's a small subset of
  // pointer typed non-type-template parameters which can't be handled it
  // doesn't seem high priority.
  f2<decltype(nullptr), nullptr>();
  // CHECK: !DISubprogram(name: "f2<std::nullptr_t, nullptr>",

  // These larger constants are encoded as data blocks which makes them a bit
  // harder to re-render. I think they might be missing sign information, or at
  // maybe it's just a question of doing APInt things to render such large
  // values. Punting on this for now.
  f2<__int128, ((__int128)9223372036854775807) * 2>();
  // CHECK: !DISubprogram(name: "f2<__int128, (__int128)18446744073709551614>",

  t2().operator t1<int>();
  // FIXME: This should be something like "operator t1<int><float>"
  // CHECK: !DISubprogram(name: "operator t1<float>",

  // Function pointer non-type-template parameters currently don't get any DWARF
  // value (GCC doesn't provide one either) and even if there was a value, if
  // it's like variable/pointer non-type template parameters, it couldn't be
  // rebuilt anyway (see the note above for details on that) so we don't have to
  // worry about seeing conversion operators as parameters to other templates.

  f3<t1>();
  // CHECK: !DISubprogram(name: "_STNf3|<t1>",
}
