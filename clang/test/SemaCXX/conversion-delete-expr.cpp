// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// Test1
struct B {
  operator char *(); // expected-note {{candidate function}}
};

struct D : B {
  operator int *(); // expected-note {{candidate function}}
};

void f (D d)
{
   delete d; // expected-error {{ambiguous conversion of delete expression of type 'D' to a pointer}}
}

// Test2
struct B1 {
  operator int *();
};

struct D1 : B1 {
  operator int *();
};

void f1 (D1 d)
{
   delete d;
}

// Test3
struct B2 {
  operator const int *();	// expected-note {{candidate function}}
};

struct D2 : B2 {
  operator int *();	// expected-note {{candidate function}}
};

void f2 (D2 d)
{
   delete d; // expected-error {{ambiguous conversion of delete expression of type 'D2' to a pointer}}
}

// Test4
struct B3 {
  operator const int *();	// expected-note {{candidate function}}
};

struct A3 {
  operator const int *();	// expected-note {{candidate function}}
};

struct D3 : A3, B3 {
};

void f3 (D3 d)
{
   delete d; // expected-error {{ambiguous conversion of delete expression of type 'D3' to a pointer}}
}

// Test5
struct X {
   operator int();
   operator int*();
};

void f4(X x) { delete x; delete x; }

// Test6
struct X1 {
   operator int();
   operator int*();
   template<typename T> operator T*() const; // converts to any pointer!
};

void f5(X1 x) { delete x; }  // OK. In selecting a conversion to pointer function, template convesions are skipped.

// Test7
struct Base {
   operator int*();	
};

struct Derived : Base {
   // not the same function as Base's non-const operator int()
   operator int*() const;
};

void foo6(const Derived cd, Derived d) {
	// overload resolution selects Derived::operator int*() const;
	delete cd;
	delete d;	
}

// Test8
struct BB {
   template<typename T> operator T*() const;
};

struct DD : BB {
   template<typename T> operator T*() const; // hides base conversion
   operator int *() const;
};

void foo7 (DD d)
{
        // OK. In selecting a conversion to pointer function, template convesions are skipped.
	delete d;
}
