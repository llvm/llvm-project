// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
//
// XFAIL: *
//
// Issue: Conditional return with multiple destructors in different scopes
//
// When a function has:
// - Multiple local variables with destructors in different scopes
// - A conditional return statement
// - Return value containing a member with a destructor
// CIR fails to properly manage the cleanup scope stack for all destructors
// that need to run on each exit path.

class a {
public:
  ~a();
};
struct b {
  a c;
};
b fn1(bool e) {
  a d;
  b f;
  if (e) {
    a d;
    return f;
  }
}
