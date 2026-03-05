// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
//
// XFAIL: *
//
// Issue: Virtual base class constructor call
//
// When a derived class constructor explicitly calls a virtual base class constructor,
// CIR fails during code generation. Virtual base class constructors require special
// handling as they are initialized by the most derived class, not intermediate classes.

class a {};
class b : virtual a {};
class c : b {
public:
  c() : b() {}
};
void d() { c e; }
