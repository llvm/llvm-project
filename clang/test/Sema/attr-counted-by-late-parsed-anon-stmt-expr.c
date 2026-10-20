// RUN: %clang_cc1 -fsyntax-only -verify=expected,eager %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify=expected,late %s

// An unnamed record followed by `;` hands its late-parsed type attributes to
// the enclosing record, since as an anonymous member its count may be a field
// of the enclosing record. Only do that for a real member: an unnamed struct
// declared in a statement expression inside a member is not one, and used to
// have its attributes resolved in the enclosing record's scope, where its own
// fields aren't visible.

#define __counted_by(f) __attribute__((counted_by(f)))

// A real anonymous member still sees the enclosing record's fields.
struct anon_member {
  struct {
    // eager-error@+1{{use of undeclared identifier 'n'}}
    int *__counted_by(n) p;
  };
  int n;
};

void stmt_expr(void) {
  struct S {
    int m;
    // expected-warning@+2{{declaration does not declare anything}}
    // expected-warning@+1{{variable length array folded to constant array as an extension}}
    char a[({ struct { int n; int *__counted_by(n) p; }; 1; })];
  };
}
