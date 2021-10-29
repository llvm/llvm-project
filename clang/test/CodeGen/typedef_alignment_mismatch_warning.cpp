// RUN: %clang_cc1 %s -fsyntax-only -verify -fdata-sections -fcolor-diagnostics

// Warn for any function that
//   * takes a pointer or a reference to an object (including "this" pointer),
//   * that was defined for aligned object,
//   * but is called with a pointer or reference to a less aligned object.
// Attributes on using declarations are ignored.

typedef __attribute__((aligned(2))) int Aligned2Int;
typedef __attribute__((aligned(8))) int Aligned8Int;

typedef __SIZE_TYPE__ size_t;

// Normal function calls
void f_takes_val(int i) {}
void f_takes_ptr(int *p) {}
void f_takes_ref(int &p) {}

void test0() {
  Aligned2Int xwarn;
  Aligned8Int xok;
  f_takes_val(xwarn);
  f_takes_ptr(&xwarn); // expected-warning {{passing 2-byte aligned argument to 4-byte aligned parameter 1 of 'f_takes_ptr' may result in an unaligned pointer access}}
  f_takes_ref(xwarn);  // expected-warning {{passing 2-byte aligned argument to 4-byte aligned parameter 1 of 'f_takes_ref' may result in an unaligned pointer access}}
  f_takes_val(xok);
  f_takes_ptr(&xok);
  f_takes_ref(xok);
}

// Constructor expects to work on an 8 byte aligned type, but is called on a (potentially) 4 byte aligned object.
void test1() {
  struct StructAligned8 {
    Aligned8Int Aligned8Member;
    StructAligned8(int whatever) : Aligned8Member(whatever) {}
  };
  typedef __attribute__((aligned(4))) StructAligned8 TypedefAligned4;
  using UsingAligned4 = __attribute__((aligned(4))) StructAligned8;

  StructAligned8 SA8(11);
  TypedefAligned4 TA4(11); // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'StructAligned8' may result in an unaligned pointer access}}
  UsingAligned4 UA4(11);
}

// Same as above, should still trigger since passing by value is irrelevant
void test1_byvalue() {
  struct StructAligned8 {
    int __attribute__((aligned(8))) Aligned8Member;
    int Irrelevant;
    StructAligned8(Aligned8Int arg) : Aligned8Member(arg) {}
  };
  typedef __attribute__((aligned(4))) StructAligned8 TypedefAligned4;
  using UsingAligned4 = __attribute__((aligned(4))) StructAligned8;

  Aligned8Int o{0};
  StructAligned8 SA8(o);
  TypedefAligned4 TA4(o); // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'StructAligned8' may result in an unaligned pointer access}}
  UsingAligned4 UA4(o);
}

// This example uses a function call trigger
void test2() {
  struct StructAligned8 {
    int __attribute__((aligned(8))) Aligned8Member;
    int Irrelevant;
  };
  auto assignment_function = [](StructAligned8 &S, Aligned8Int arg) {
    S.Aligned8Member = arg;
  };
  typedef __attribute__((aligned(4))) StructAligned8 TypedefAligned4;
  using UsingAligned4 = __attribute__((aligned(4))) StructAligned8;

  StructAligned8 SA8;
  TypedefAligned4 TA4;          // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'StructAligned8' may result in an unaligned pointer access}}
  assignment_function(TA4, 11); // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 1 of 'operator()' may result in an unaligned pointer access}}
  UsingAligned4 UA4;
  assignment_function(UA4, 11);
}

// Same as above, but should not trigger as passed by value
void test2_byvalue() {
  struct StructAligned8 {
    int __attribute__((aligned(8))) Aligned8Member;
    int Irrelevant;
  };
  auto assignment_function = [](StructAligned8 S, Aligned8Int arg) {
    S.Aligned8Member = arg;
  };
  typedef __attribute__((aligned(4))) StructAligned8 TypedefAligned4;
  using UsingAligned4 = __attribute__((aligned(4))) StructAligned8;

  Aligned8Int o{0};
  StructAligned8 SA8;
  TypedefAligned4 TA4; // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'StructAligned8' may result in an unaligned pointer access}}
  assignment_function(TA4, o);
  UsingAligned4 UA4;
  assignment_function(UA4, o);
}

void test4() {
  struct StructWithPackedMember {
    int PackedMember __attribute__((packed));
  } SWPM;

  // Explicitly taking the address of an unaligned member causes a warning
  (void)&SWPM.PackedMember; // expected-warning {{taking address of packed member 'PackedMember' of class or structure 'StructWithPackedMember' may result in an unaligned pointer value}}
}

// Aligned attribute on struct itself
void test5() {
  struct __attribute__((aligned(8))) StructAligned8 {
    int Aligned8Member;
    int Irrelevant;
    StructAligned8(int i) : Aligned8Member(i) {}
  };
  typedef __attribute__((aligned(4))) StructAligned8 TypedefAligned4;
  using UsingAligned4 = __attribute__((aligned(4))) StructAligned8;

  StructAligned8 SA8(11);
  TypedefAligned4 TA4(11); // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'StructAligned8' may result in an unaligned pointer access}}
  UsingAligned4 UA4(11);
}

// Via function pointer
void test6() {
  struct __attribute__((aligned(8))) StructAligned8 {
    int Aligned8Member;
    StructAligned8(int i) : Aligned8Member(i) {}
  };

  auto assignment_function_ref = [](StructAligned8 &S) {
    S.Aligned8Member = 42;
  };
  auto assignment_function_ptr = [](StructAligned8 *S) {
    S->Aligned8Member = 42;
  };

  void (*RefFnPtr)(StructAligned8 &) = assignment_function_ref;
  void (*PtrFnPtr)(StructAligned8 *) = assignment_function_ptr;

  typedef __attribute__((aligned(4))) StructAligned8 TypedefAligned4;
  using UsingAligned4 = __attribute__((aligned(4))) StructAligned8;

  StructAligned8 SA8(11);
  RefFnPtr(SA8);
  PtrFnPtr(&SA8);
  TypedefAligned4 TA4(11); // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'StructAligned8' may result in an unaligned pointer access}}
  RefFnPtr(TA4);           // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 1 of 'RefFnPtr' may result in an unaligned pointer access}}
  PtrFnPtr(&TA4);          // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 1 of 'PtrFnPtr' may result in an unaligned pointer access}}
  UsingAligned4 UA4(11);
  RefFnPtr(UA4);
  PtrFnPtr(&UA4);
}

// Member function
void test7() {
  struct __attribute__((aligned(8))) StructAligned8 {
    int Aligned8Member;
    StructAligned8(int i) : Aligned8Member(i) {}
    void memberFnAssignment() {
      Aligned8Member = 42;
    }
  };

  typedef __attribute__((aligned(4))) StructAligned8 TypedefAligned4;
  using UsingAligned4 = __attribute__((aligned(4))) StructAligned8;

  StructAligned8 SA8(11);
  SA8.memberFnAssignment();
  TypedefAligned4 TA4(11);  // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'StructAligned8' may result in an unaligned pointer access}}
  TA4.memberFnAssignment(); // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'memberFnAssignment' may result in an unaligned pointer access}}
  UsingAligned4 UA4(11);
  UA4.memberFnAssignment();

  // Check access through pointer
  StructAligned8 *SA8ptr;
  SA8ptr->memberFnAssignment();
  TypedefAligned4 *TA4ptr;
  TA4ptr->memberFnAssignment(); // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'memberFnAssignment' may result in an unaligned pointer access}}
  UsingAligned4 *UA4ptr;
  UA4ptr->memberFnAssignment();
}

// Member binary and unary operator
void test8() {
  struct __attribute__((aligned(8))) StructAligned8 {
    int Aligned8Member;
    StructAligned8(int i) : Aligned8Member(i) {}
    StructAligned8 operator+(StructAligned8 &other) {
      return {other.Aligned8Member + Aligned8Member};
    }
    StructAligned8 operator-(StructAligned8 *other) {
      return {other->Aligned8Member + Aligned8Member};
    }
    StructAligned8 &operator++() {
      Aligned8Member++;
      return *this;
    }
    StructAligned8 &operator--() {
      Aligned8Member--;
      return *this;
    }
  };

  typedef __attribute__((aligned(4))) StructAligned8 TypedefAligned4;
  using UsingAligned4 = __attribute__((aligned(4))) StructAligned8;

  StructAligned8 SA8a(11);
  StructAligned8 SA8b(11);
  auto SA8c = SA8a + SA8b;
  auto SA8d = SA8a - &SA8b;
  ++SA8c;
  --SA8d;
  TypedefAligned4 TA8a(11);            // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'StructAligned8' may result in an unaligned pointer access}}
  TypedefAligned4 TA8b(11);            // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'StructAligned8' may result in an unaligned pointer access}}
  TypedefAligned4 TA8c = TA8a + TA8b;  // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'operator+' may result in an unaligned pointer access}}
                                       // expected-warning@-1 {{passing 4-byte aligned argument to 8-byte aligned parameter 1 of 'operator+' may result in an unaligned pointer access}}
                                       // expected-warning@-2 {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'StructAligned8' may result in an unaligned pointer access}}
  TypedefAligned4 TA8d = TA8a - &TA8b; // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'operator-' may result in an unaligned pointer access}}
                                       // expected-warning@-1 {{passing 4-byte aligned argument to 8-byte aligned parameter 1 of 'operator-' may result in an unaligned pointer access}}
                                       // expected-warning@-2 {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'StructAligned8' may result in an unaligned pointer access}}
  ++TA8d;                              // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'operator++' may result in an unaligned pointer access}}
  --TA8c;                              // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'operator--' may result in an unaligned pointer access}}
  UsingAligned4 UA8a(11);
  UsingAligned4 UA8b(11);
  auto UA8c = UA8a + UA8b;
  auto UA8d = UA8a - &UA8b;
  ++UA8c;
  --UA8d;

  // Bonus
  auto bonus1 = TA8a + SA8b;  // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'operator+' may result in an unaligned pointer access}}
  auto bonus2 = SA8a + TA8b;  // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 1 of 'operator+' may result in an unaligned pointer access}}
  auto bonus3 = TA8a - &SA8b; // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'operator-' may result in an unaligned pointer access}}
  auto bonus4 = SA8a - &TA8b; // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 1 of 'operator-' may result in an unaligned pointer access}}
}

// Static binary operator
struct __attribute__((aligned(8))) test9Struct {
  int Aligned8Member;
  test9Struct(int i) : Aligned8Member(i) {}
};
test9Struct operator+(test9Struct &a, test9Struct &b) {
  return {a.Aligned8Member + b.Aligned8Member};
}
void test9() {

  typedef __attribute__((aligned(4))) test9Struct TypedefAligned4;
  using UsingAligned4 = __attribute__((aligned(4))) test9Struct;

  test9Struct SA8a(11);
  test9Struct SA8b(11);
  auto SA8c = SA8a + SA8b;
  TypedefAligned4 TA8a(11); // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'test9Struct' may result in an unaligned pointer access}}
  TypedefAligned4 TA8b(11); // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'test9Struct' may result in an unaligned pointer access}}
  auto TA8c = TA8a + TA8b;  // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 1 of 'operator+' may result in an unaligned pointer access}}
                            // expected-warning@-1 {{passing 4-byte aligned argument to 8-byte aligned parameter 2 of 'operator+' may result in an unaligned pointer access}}
  UsingAligned4 UA8a(11);
  UsingAligned4 UA8b(11);
  auto UA8c = UA8a + UA8b;

  // Bonus
  auto bonus1 = TA8a + SA8b; // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 1 of 'operator+' may result in an unaligned pointer access}}
  auto bonus2 = SA8a + TA8b; // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 2 of 'operator+' may result in an unaligned pointer access}}
}

// Operator new and placement new
void test10() {
  struct __attribute__((aligned(8))) StructAligned8 {
    int Aligned8Member;
    StructAligned8(int i) : Aligned8Member(i) {}
    void *operator new(size_t count) { return (void *)0x123456; }
    void *operator new(size_t count, void *p) { return p; }
  };

  typedef __attribute__((aligned(4))) StructAligned8 TypedefAligned4;
  using UsingAligned4 = __attribute__((aligned(4))) StructAligned8;

  auto *SA8ptr = new StructAligned8(11);
  new (SA8ptr) StructAligned8(12);
  auto *TA4ptr = new TypedefAligned4(11); // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'StructAligned8' may result in an unaligned pointer access}}
  new (TA4ptr) TypedefAligned4(12);       // expected-warning {{passing 4-byte aligned argument to 8-byte aligned parameter 'this' of 'StructAligned8' may result in an unaligned pointer access}}
  auto *UA4ptr = new UsingAligned4(11);
  new (UA4ptr) UsingAligned4(12);
}

void testFunctionPointerArray(void (*fptr[10])(Aligned8Int *), Aligned2Int* src) {
  fptr[0](src); // expected-warning {{passing 2-byte aligned argument to 8-byte aligned parameter 1 may result in an unaligned pointer access}}
}
