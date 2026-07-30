// RUN: %clang_cc1 -fsyntax-only -verify %s
struct A {};
struct B : public A {};             // Single public base.
struct C1 : public virtual B {};    // Single virtual base.
struct C2 : public virtual B {};
struct D : public C1, public C2 {}; // Diamond
struct E : private A {};            // Single private base. expected-note 3 {{declared private here}}
struct F : public C1 {};            // Single path to B with virtual.
struct G1 : public B {};
struct G2 : public B {};
struct H : public G1, public G2 {}; // Ambiguous path to B.
struct I;                           // Incomplete.  expected-note {{'I' is incomplete}}
struct J;                           // Incomplete.  expected-note {{'J' is incomplete}}

enum Enum { En1, En2 };
enum Onom { On1, On2 };

struct Co1 { operator int(); };
struct Co2 { Co2(int); };
struct Co3 { };
struct Co4 { Co4(Co3); operator Co3(); };

// Explicit implicits
void t_529_2()
{
  int i = 1;
  (void)static_cast<float>(i);
  double d = 1.0;
  (void)static_cast<float>(d);
  (void)static_cast<int>(d);
  (void)static_cast<char>(i);
  (void)static_cast<unsigned long>(i);
  (void)static_cast<int>(En1);
  (void)static_cast<double>(En1);
  (void)static_cast<int&>(i);
  (void)static_cast<const int&>(i);

  int ar[1];
  (void)static_cast<const int*>(ar);
  (void)static_cast<void (*)()>(t_529_2);

  (void)static_cast<void*>(0);
  (void)static_cast<void*>((int*)0);
  (void)static_cast<volatile const void*>((const int*)0);
  (void)static_cast<A*>((B*)0);
  (void)static_cast<A&>(*((B*)0)); // expected-warning {{binding dereferenced null pointer to reference has undefined behavior}}
  (void)static_cast<const B*>((C1*)0);
  (void)static_cast<B&>(*((C1*)0)); // expected-warning {{binding dereferenced null pointer to reference has undefined behavior}}
  (void)static_cast<A*>((D*)0);
  (void)static_cast<const A&>(*((D*)0)); // expected-warning {{binding dereferenced null pointer to reference has undefined behavior}}
  (void)static_cast<int B::*>((int A::*)0);
  (void)static_cast<void (B::*)()>((void (A::*)())0);

  (void)static_cast<int>(Co1());
  (void)static_cast<Co2>(1);
  (void)static_cast<Co3>(static_cast<Co4>(Co3()));

  // Bad code below

  (void)static_cast<void*>((const int*)0); // expected-error {{static_cast from 'const int *' to 'void *' is not allowed}}
  (void)static_cast<A*>((E*)0); // expected-error {{cannot cast 'E' to its private base class 'A'}}
  (void)static_cast<A*>((H*)0); // expected-error {{ambiguous conversion}}
  (void)static_cast<int>((int*)0); // expected-error {{static_cast from 'int *' to 'int' is not allowed}}
  (void)static_cast<A**>((B**)0); // expected-error {{static_cast from 'B **' to 'A **' is not allowed}}
  (void)static_cast<char&>(i); // expected-error {{non-const lvalue reference to type 'char' cannot bind to a value of unrelated type 'int'}}
}

// Anything to void
void t_529_4()
{
  static_cast<void>(1);
  static_cast<void>(t_529_4);
}

// Static downcasts
void t_529_5_8()
{
  (void)static_cast<B*>((A*)0);
  (void)static_cast<B&>(*((A*)0));
  (void)static_cast<const G1*>((A*)0);
  (void)static_cast<const G1&>(*((A*)0));

  // Bad code below

  (void)static_cast<C1*>((A*)0); // expected-error {{cannot cast 'A *' to 'C1 *' via virtual base 'B'}}
  (void)static_cast<C1&>(*((A*)0)); // expected-error {{cannot cast 'A' to 'C1 &' via virtual base 'B'}}
  (void)static_cast<D*>((A*)0); // expected-error {{cannot cast 'A *' to 'D *' via virtual base 'B'}}
  (void)static_cast<D&>(*((A*)0)); // expected-error {{cannot cast 'A' to 'D &' via virtual base 'B'}}
  (void)static_cast<B*>((const A*)0); // expected-error {{static_cast from 'const A *' to 'B *' casts away qualifiers}}
  (void)static_cast<B&>(*((const A*)0)); // expected-error {{static_cast from 'const A' to 'B &' casts away qualifiers}}
  (void)static_cast<E*>((A*)0); // expected-error {{cannot cast private base class 'A' to 'E'}}
  (void)static_cast<E&>(*((A*)0)); // expected-error {{cannot cast private base class 'A' to 'E'}}
  (void)static_cast<H*>((A*)0); // expected-error {{ambiguous cast from base 'A' to derived 'H':\n    A -> B -> G1 -> struct H\n    struct A -> B -> G2 -> struct H}}
  (void)static_cast<H&>(*((A*)0)); // expected-error {{ambiguous cast from base 'A' to derived 'H':\n    A -> B -> G1 -> struct H\n    A -> B -> G2 -> struct H}}
  (void)static_cast<E*>((B*)0); // expected-error {{static_cast from 'B *' to 'E *', which are not related by inheritance, is not allowed}}
  (void)static_cast<E&>(*((B*)0)); // expected-error {{non-const lvalue reference to type 'E' cannot bind to a value of unrelated type 'B'}}


  (void)static_cast<E*>((J*)0); // expected-error {{static_cast from 'J *' to 'E *', which are not related by inheritance, is not allowed}}
  (void)static_cast<I*>((B*)0); // expected-error {{static_cast from 'B *' to 'I *', which are not related by inheritance, is not allowed}}

  // TODO: Test inaccessible base in context where it's accessible, i.e.
  // member function and friend.

  // TODO: Test DR427. This requires user-defined conversions, though.
}

// Enum conversions
void t_529_7()
{
  (void)static_cast<Enum>(1);
  (void)static_cast<Enum>(1.0);
  (void)static_cast<Onom>(En1);

  // Bad code below

  (void)static_cast<Enum>((int*)0); // expected-error {{static_cast from 'int *' to 'Enum' is not allowed}}
}

// Void pointer to object pointer
void t_529_10()
{
  (void)static_cast<int*>((void*)0);
  (void)static_cast<const A*>((void*)0);

  // Bad code below

  (void)static_cast<int*>((const void*)0); // expected-error {{static_cast from 'const void *' to 'int *' casts away qualifiers}}
  (void)static_cast<void (*)()>((void*)0); // expected-error {{static_cast from 'void *' to 'void (*)()' is not allowed}}
}

// Member pointer upcast.
void t_529_9()
{
  (void)static_cast<int A::*>((int B::*)0);

  // Bad code below
  (void)static_cast<int A::*>((int H::*)0); // expected-error {{ambiguous conversion from pointer to member of derived class 'H' to pointer to member of base class 'A':}}
  (void)static_cast<int A::*>((int F::*)0); // expected-error {{conversion from pointer to member of class 'F' to pointer to member of class 'A' via virtual base 'B' is not allowed}}
  (void)static_cast<int I::*>((int J::*)0); // expected-error {{static_cast from 'int J::*' to 'int I::*' is not allowed}}
}

// PR 5261 - static_cast should instantiate template if possible
namespace pr5261 {
  struct base {};
  template<typename E> struct derived : public base {};
  template<typename E> struct outer {
    base *pb;
    ~outer() { (void)static_cast<derived<E>*>(pb); }
  };
  outer<int> EntryList;
}


// Initialization by constructor
struct X0;

struct X1 {
  X1();
  X1(X1&);
  X1(const X0&);
  
  operator X0() const;
};

struct X0 { };

void test_ctor_init() {
  (void)static_cast<X1>(X1());
}

// Casting away constness
struct X2 {
};

struct X3 : X2 {
};

struct X4 {
  typedef const X3 X3_typedef;
  
  void f() const {
    (void)static_cast<X3_typedef*>(x2);
  }
  
  const X2 *x2;
};

// PR5897 - accept static_cast from const void* to const int (*)[1].
void PR5897() { (void)static_cast<const int(*)[1]>((const void*)0); }

namespace PR6072 {
  struct A { }; 
  struct B : A { void f(int); void f(); };  // expected-note 2{{candidate function}}
  struct C : B { };
  struct D { };

  void f() {
    (void)static_cast<void (A::*)()>(&B::f);
    (void)static_cast<void (B::*)()>(&B::f);
    (void)static_cast<void (C::*)()>(&B::f);
    (void)static_cast<void (D::*)()>(&B::f); // expected-error-re{{address of overloaded function 'f' cannot be static_cast to type 'void (D::*)(){{( __attribute__\(\(thiscall\)\))?}}'}}
  }
}
