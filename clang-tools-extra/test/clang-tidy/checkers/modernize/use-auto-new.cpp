// RUN: %check_clang_tidy %s modernize-use-auto %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-auto.MinTypeNameLength: '0'}}" \
// RUN:   -- -frtti

class MyType {};

class MyDerivedType : public MyType {};

// FIXME: the replacement sometimes results in two consecutive spaces after
// the word 'auto' (due to the presence of spaces at both sides of '*').
void auto_new() {
  MyType *a_new = new MyType();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with new
  // CHECK-FIXES: auto *a_new = new MyType();

  static MyType *a_static = new MyType();
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use auto when initializing with new
  // CHECK-FIXES: static auto *a_static = new MyType();

  long long *ll = new long long();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with new
  // CHECK-FIXES: auto *ll = new long long();

  MyType *derived = new MyDerivedType();

  void *vd = new MyType();

  // CV-qualifier tests.
  //
  // NOTE : the form "type const" is expected here because of a deficiency in
  // TypeLoc where CV qualifiers are not considered part of the type location
  // info. That is, all that is being replaced in each case is "MyType *" and
  // not "MyType * const".
  static MyType * const d_static = new MyType();
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use auto when initializing with new
  // CHECK-FIXES: static auto * const d_static = new MyType();

  MyType * const a_const = new MyType();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with new
  // CHECK-FIXES: auto * const a_const = new MyType();

  MyType * volatile vol = new MyType();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with new
  // CHECK-FIXES: auto * volatile vol = new MyType();

  struct SType {} *stype = new SType;

  int (**func)(int, int) = new (int(*[5])(int,int));

  int *array = new int[5];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with new
  // CHECK-FIXES: auto *array = new int[5];

  MyType *ptr(new MyType);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with new
  // CHECK-FIXES: auto *ptr(new MyType);

  MyType *ptr2{new MyType};

  {
    // Test for declaration lists.
    MyType *a = new MyType(), *b = new MyType(), *c = new MyType();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use auto when initializing with new
    // CHECK-FIXES: auto *a = new MyType(), *b = new MyType(), *c = new MyType();

    // Non-initialized declaration should not be transformed.
    MyType *d = new MyType(), *e;

    MyType **f = new MyType*(), **g = new MyType*();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use auto when initializing with new
    // CHECK-FIXES: auto **f = new MyType*(), **g = new MyType*();

    // Mismatching types in declaration lists should not be transformed.
    MyType *h = new MyType(), **i = new MyType*();

    // '*' shouldn't be removed in case of mismatching types with multiple
    // declarations.
    MyType *j = new MyType(), *k = new MyType(), **l = new MyType*();
  }

  {
    // Test for typedefs.
    typedef int * int_p;
    // CHECK-FIXES: typedef int * int_p;

    int_p a = new int;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use auto when initializing with new
    // CHECK-FIXES: auto a = new int;
    int_p *b = new int*;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use auto when initializing with new
    // CHECK-FIXES: auto *b = new int*;

    // Test for typedefs in declarations lists.
    int_p c = new int, d = new int;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use auto when initializing with new
    // CHECK-FIXES: auto c = new int, d = new int;

    // Different types should not be transformed.
    int_p e = new int, *f = new int_p;

    int_p *g = new int*, *h = new int_p;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use auto when initializing with new
    // CHECK-FIXES: auto *g = new int*, *h = new int_p;
  }

  // Don't warn when 'auto' is already being used.
  auto aut = new MyType();
  auto *paut = new MyType();
  const auto *pcaut = new MyType();
}
