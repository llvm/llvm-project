// RUN: %clang_cc1 -fsyntax-only -verify=expected,precxx17 %std_cxx98-14 %s
// RUN: %clang_cc1 -fsyntax-only -verify %std_cxx17- %s
template<typename T>
struct X1 {
  static void member() { T* x = 1; } // precxx17-error{{cannot initialize a variable of type 'int *' with an rvalue of type 'int'}}
};

template<void(*)()> struct instantiate { };

template<typename T>
struct X2 {
  typedef instantiate<&X1<int>::member> i; // precxx17-note{{in instantiation of}}
};

X2<int> x;

template <class T, class A> class C {
public:
  int i;
  void f(T &t) {
    T *q = new T();
    t.T::~T();
    q->~T();
    // expected-error@+2 {{'int' is not a class, namespace, or enumeration}}
    // expected-error@+1 {{no member named '~Colors' in 'Colors'}}
    q->A::~A();
    // expected-error@+2 {{no member named '~int' in 'Q'}}
    // expected-error@+1 {{no member named '~Colors' in 'Q'}}
    q->~A();

    delete q;
  }
};

class Q {
public:
  Q() {}
  ~Q() {}
};

enum Colors {red, green, blue};

C<Q, int> dummy;
C<Q, Colors> dummyColors;
int main() {
  Q qinst;
  // expected-note@+1 {{in instantiation of member function 'C<Q, int>::f' requested here}}
  dummy.f(qinst);
  // expected-note@+1 {{in instantiation of member function 'C<Q, Colors>::f' requested here}}
  dummyColors.f(qinst);
}

