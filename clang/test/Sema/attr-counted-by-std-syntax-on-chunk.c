// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c23 -fsyntax-only -fexperimental-late-parse-attributes -verify %s
// RUN: not %clang_cc1 -std=c23 -fsyntax-only -ast-dump %s 2>/dev/null | FileCheck %s
// RUN: not %clang_cc1 -std=c23 -fsyntax-only -fexperimental-late-parse-attributes -ast-dump %s 2>/dev/null | FileCheck %s

// A [[]] counted_by-family attribute on a declarator chunk is a type
// attribute. It applies to a field's pointer, and is rejected rather than
// dropped elsewhere. It isn't late-parsed, so the late-parsing mode handles it
// the same way.

void f(int n, int *[[clang::counted_by(n)]] p); // expected-error{{'clang::counted_by' attribute cannot be applied to types}}

int gn;
int *[[clang::sized_by(gn)]] gp; // expected-error{{'clang::sized_by' attribute cannot be applied to types}}

// After the type specifiers, it applies to the type specifiers' type, and is
// rejected the same way outside of a field. It's diagnosed once even though
// every declarator shares it.
void g(int n, int [[clang::counted_by(n)]] *p); // expected-error{{'clang::counted_by' attribute cannot be applied to types}}
int [[clang::counted_by(gn)]] *ga, *gb; // expected-error{{'clang::counted_by' attribute cannot be applied to types}}
int [[clang::counted_by(gn)]] gi; // expected-error{{'clang::counted_by' attribute cannot be applied to types}}
typedef int [[clang::counted_by(gn)]] *T; // expected-error{{'clang::counted_by' attribute cannot be applied to types}}
void h(void) {
  int [[clang::counted_by(gn)]] *l; // expected-error{{'clang::counted_by' attribute cannot be applied to types}}
  (void)sizeof(int [[clang::counted_by(gn)]] *); // expected-error{{'clang::counted_by' attribute cannot be applied to types}}
}

struct S {
  int n;
  // CHECK: FieldDecl {{.*}} p 'int * __counted_by_or_null(n)':'int *'
  int *[[clang::counted_by_or_null(n)]] p;
  int *[[clang::counted_by(n)]] *pp; // expected-error{{'counted_by' attribute on nested pointer type is not allowed}}
};

// Declarators nest, so the position of an attribute is measured from the chunk
// it's on. An attribute anywhere in the return type of a function a pointer
// points to is on a return type, however deeply that function is nested.
struct ReturnTypes {
  int n;
  // A function pointer returning a function pointer returning a pointer.
  // expected-error@+1{{'counted_by' attribute on a function return type is not yet supported}}
  int *[[clang::counted_by(n)]] (*(*fpfp)(void))(void);
  // The same, returning a pointer to a counted array.
  // expected-error@+1{{'counted_by' attribute on a function return type is not yet supported}}
  int (*(*(*fpfp_arr)(void))(void))[] [[clang::counted_by(n)]];
  // The same, with a fixed-size array. The position is checked before whether
  // the array can be counted.
  // expected-error@+1{{'counted_by' attribute on a function return type is not yet supported}}
  int (*(*(*fpfp_arr4)(void))(void))[4] [[clang::counted_by(n)]];
  // A pointer to an array of function pointers that return a pointer to a
  // counted array.
  // expected-error@+1{{'counted_by' attribute on a function return type is not yet supported}}
  int (*(*(*pafp)[4])(void))[] [[clang::counted_by(n)]];
};

// An array behind a pointer is nested.
struct PointerToArray {
  int n;
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  int (*pa)[] [[clang::counted_by(n)]];
};

// A flexible array member is the one array that takes the attribute, with or
// without parentheses around its name.
struct FlexibleArrayMember {
  int n;
  // CHECK: FieldDecl {{.*}} fam 'int[] __counted_by(n)':'int[]'
  int fam[] [[clang::counted_by(n)]];
};

struct ParenthesizedFlexibleArrayMember {
  int n;
  // CHECK: FieldDecl {{.*}} fam 'int[] __counted_by(n)':'int[]'
  int (fam)[] [[clang::counted_by(n)]];
};
