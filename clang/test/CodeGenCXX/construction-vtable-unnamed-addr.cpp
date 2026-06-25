// C has no key function, so its primary vtable and construction vtables have
// weak (vague) linkage. On a target that may duplicate vtables (Apple Mach-O)
// such weak vtables are marked unnamed_addr; on other targets they are not.
// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -std=c++11 -emit-llvm -o - | FileCheck %s --check-prefix=DARWIN
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -std=c++11 -emit-llvm -o - | FileCheck %s --check-prefix=LINUX

struct V { virtual void f() {} virtual ~V() {} };
struct A : virtual V { virtual void a() {} };
struct B : virtual V { virtual void b() {} };
struct C : A, B { virtual void c() {} };

C *make() { return new C(); }

// C's primary vtable and the construction vtables for its A and B bases are
// weak, so they are marked unnamed_addr on Mach-O but not on Linux.
// DARWIN-DAG: @_ZTV1C = linkonce_odr unnamed_addr constant
// DARWIN-DAG: @_ZTC1C0_1A = linkonce_odr unnamed_addr constant
// DARWIN-DAG: @_ZTC1C8_1B = linkonce_odr unnamed_addr constant

// LINUX-DAG: @_ZTV1C = linkonce_odr constant
// LINUX-DAG: @_ZTC1C0_1A = linkonce_odr constant
// LINUX-DAG: @_ZTC1C8_1B = linkonce_odr constant
