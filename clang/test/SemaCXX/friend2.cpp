// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

// If a friend function is defined in several non-template classes,
// it is an error.

void func1(int);
struct C1a {
  friend void func1(int) {}  // expected-note{{previous definition is here}}
};
struct C1b {
  friend void func1(int) {}  // expected-error{{redefinition of 'func1'}}
};


// If a friend function is defined in both non-template and template
// classes it is an error only if the template is instantiated.

void func2(int);
struct C2a {
  friend void func2(int) {}
};
template<typename T> struct C2b {
  friend void func2(int) {}
};

void func3(int);
struct C3a {
  friend void func3(int) {}  // expected-note{{previous definition is here}}
};
template<typename T> struct C3b {
  friend void func3(int) {}  // expected-error{{redefinition of 'func3'}}
};
C3b<long> c3;  // expected-note{{in instantiation of template class 'C3b<long>' requested here}}


// If a friend function is defined in several template classes it is an error
// only if several templates are instantiated.

void func4(int);
template<typename T> struct C4a {
  friend void func4(int) {}
};
template<typename T> struct C4b {
  friend void func4(int) {}
};


void func5(int);
template<typename T> struct C5a {
  friend void func5(int) {}
};
template<typename T> struct C5b {
  friend void func5(int) {}
};
C5a<long> c5a;

void func6(int);
template<typename T> struct C6a {
  friend void func6(int) {}  // expected-note{{previous definition is here}}
};
template<typename T> struct C6b {
  friend void func6(int) {}  // expected-error{{redefinition of 'func6'}}
};
C6a<long> c6a;
C6b<int*> c6b;  // expected-note{{in instantiation of template class 'C6b<int *>' requested here}}

void func7(int);
template<typename T> struct C7 {
  friend void func7(int) {}  // expected-error{{redefinition of 'func7'}}
                             // expected-note@-1{{previous definition is here}}
};
C7<long> c7a;
C7<int*> c7b;  // expected-note{{in instantiation of template class 'C7<int *>' requested here}}


// Even if clases are not instantiated and hence friend functions defined in them are not
// available, their declarations can be checked.

void func8(int);  // expected-note{{previous declaration is here}}
template<typename T> struct C8a {
  friend long func8(int);  // expected-error{{functions that differ only in their return type cannot be overloaded}}
};

void func9(int);  // expected-note{{previous declaration is here}}
template<typename T> struct C9a {
  friend int func9(int);  // expected-error{{functions that differ only in their return type cannot be overloaded}}
};

void func10(int);  // expected-note{{previous declaration is here}}
template<typename T> struct C10a {
  friend int func10(int);  // expected-error{{functions that differ only in their return type cannot be overloaded}}
};

void func_11();  // expected-note{{previous declaration is here}}
template<typename T> class C11 {
  friend int func_11();  // expected-error{{functions that differ only in their return type cannot be overloaded}}
};

void func_12(int x);  // expected-note{{previous declaration is here}}
template<typename T> class C12 {
  friend void func_12(int x = 0);  // expected-error{{friend declaration specifying a default argument must be the only declaration}}
};


namespace pr22307 {

struct t {
  friend int leak(t);
};

template<typename v>
struct m {
  friend int leak(t) { return sizeof(v); }  // expected-error{{redefinition of 'leak'}} expected-note{{previous definition is here}}
};

template struct m<char>;
template struct m<short>;  // expected-note{{in instantiation of template class 'pr22307::m<short>' requested here}}

int main() {
  leak(t());
}

}

namespace pr17923 {

void f(unsigned long long);

template<typename T> struct X {
  friend void f(unsigned long long) {
     T t;
  }
};

int main() { f(1234); }

}

namespace pr17923a {

int get();

template< int value >
class set {
  friend int get()
    { return value; } // return 0; is OK
};

template class set< 5 >;

int main() {
  get();
}

}

namespace pr8035 {

void Function();

int main(int argc, char* argv[]) {
  Function();
}

template <typename T>
struct Test {
  friend void Function() { }
};

template class Test<int>;

}
