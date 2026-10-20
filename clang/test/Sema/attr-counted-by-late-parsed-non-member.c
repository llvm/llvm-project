// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify %s

// Late-parsed type attributes are only supported on struct members, whose
// arguments are resolved at the record's closing brace. Declarators nested in
// a member that aren't members themselves, such as the parameters of a
// function-pointer member or a type name in `sizeof`, parse the attribute
// eagerly. Diagnostics match those without late parsing.
//
// Late parsing used to apply to these declarators too, which broke:
// - a parameter's count was looked up in struct scope at the closing brace, so
//   it missed the function's other parameters or silently bound to a field;
// - a paren that turned out to be a parameter list hit
//   `assert(LateAttrs.empty())` in ParseParenDeclarator.
// Parameters are tracked by https://github.com/llvm/llvm-project/issues/166454.

#define __counted_by(f) __attribute__((counted_by(f)))

struct fp_param_count_in_proto {
  // expected-error@+1{{'counted_by' attribute only applies to non-static data members}}
  void (*cb)(int len, int *__counted_by(len) buf);
};

struct fp_param_names_field {
  int len;
  // A late-parsed count used to bind to the field `len` with no diagnostic.
  // expected-error@+1{{use of undeclared identifier 'len'}}
  void (*cb)(int *__counted_by(len) buf);
};

struct paren_param_list {
  int n;
  // The attribute applies to the first parameter of the inner function type.
  // expected-error@+1{{use of undeclared identifier 'n'}}
  void (*fp)(int (__counted_by(n) int));
};

struct sizeof_abstract {
  int n;
  // expected-error@+1{{argument required after attribute}}
  char a[sizeof(int *(__counted_by(n)))];
};
