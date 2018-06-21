// Makes sure it doesn't crash.

// XFAIL: linux

// RUN: rm -rf %t
// RUN: %clang_cc1 %s -index-store-path %t/idx -std=c++14
// RUN: c-index-test core -print-record %t/idx | FileCheck %s

namespace crash1 {
// CHECK: [[@LINE+1]]:6 | function/C
auto getit() { return []() {}; }
}

namespace crash2 {
// CHECK: [[@LINE+2]]:7 | class(Gen)/C++ | c:@N@crash2@ST>1#T@Foo | Decl,RelChild | rel: 1
template <typename T>
class Foo; // canonical decl

// CHECK: [[@LINE+2]]:7 | class(Gen)/C++ | c:@N@crash2@ST>1#T@Foo | Def,RelChild | rel: 1
template <typename T>
class Foo {};

// CHECK: [[@LINE+2]]:8 | struct(Gen)/C++ | c:@N@crash2@ST>1#t>1#pT@Wrapper | Def,RelChild | rel: 1
template <template <typename... ARGS> class TYPE>
struct Wrapper {};

// CHECK: [[@LINE+3]]:8 | struct(Gen,TS)/C++ | c:@N@crash2@S@Wrapper>#@N@crash2@ST>1#T@Foo | Def,RelChild,RelSpecialization | rel: 2
// CHECK:	RelSpecialization | c:@N@crash2@ST>1#t>1#pT@Wrapper
template <>
struct Wrapper<Foo> {};
}
