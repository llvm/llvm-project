// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
//
// XFAIL: *
//
// Issue: Global variable with virtual method and destructor
//
// When a global variable has both:
// - A non-trivial destructor requiring registration for cleanup at program exit
// - A virtual method requiring vtable generation
// CIR fails to properly coordinate the vtable setup with destructor registration.

class a {
public:
  ~a();
  virtual char b();
} c;
