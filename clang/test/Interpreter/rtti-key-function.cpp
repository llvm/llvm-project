// REQUIRES: host-supports-jit
// UNSUPPORTED: system-windows
// RUN: cat %s | clang-repl | FileCheck %s

// The typeinfo of a class with a key function is emitted with its vtable,
// into the module of the input that defines the key function. Later inputs
// that need it must not define it again: the JIT reported
//     duplicate definition of symbol '_ZTI1A'

extern "C" int printf(const char *, ...);
namespace std { class type_info; }

struct A { virtual void f(); };
void A::f() {}
const std::type_info *t1 = &typeid(A);
const std::type_info *t2 = &typeid(A);
printf("same typeinfo: %d\n", t1 == t2);
// CHECK: same typeinfo: 1

// The key function and the use of the typeinfo in one input.
struct B { virtual void g(); };
void B::g() {} const std::type_info *t3 = &typeid(B);
const std::type_info *t4 = &typeid(B);
printf("same typeinfo: %d\n", t3 == t4);
// CHECK: same typeinfo: 1

// Without a key function, the typeinfo is emitted where it is used.
struct C { virtual void h() {} };
const std::type_info *t5 = &typeid(C);
const std::type_info *t6 = &typeid(C);
printf("same typeinfo: %d\n", t5 == t6);
// CHECK: same typeinfo: 1

%quit
