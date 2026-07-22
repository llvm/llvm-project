// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -verify %s

// The defaulted copy/move assignment operator of a union has an empty
// synthesized body -- Sema skips union fields, leaving no AST expression for
// the implied whole-object copy.  Emitting that body would silently drop the
// copy, so CIRGen reports NYI instead.

// expected-error@+1 2 {{ClangIR code gen Not Yet Implemented: defaulted union copy/move assignment operator}}
union U {
  void *p;
  int i;
};

void copy_assign(U &a, U &b) { a = b; }
void move_assign(U &a, U &b) { a = static_cast<U &&>(b); }
