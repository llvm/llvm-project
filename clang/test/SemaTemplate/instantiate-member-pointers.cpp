// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
struct Y {
  int x;
};

template<typename T>
struct X1 {
  int f(T* ptr, int T::*pm) { // expected-error{{member pointer}}
    return ptr->*pm;
  }
};

template struct X1<Y>;
template struct X1<int>; // expected-note{{instantiation}}

template<typename T, typename Class>
struct X2 {
  T f(Class &obj, T Class::*pm) { // expected-error{{to a reference}} \
                      // expected-error{{member pointer to void}}
    return obj.*pm; 
  }
};

template struct X2<int, Y>;
template struct X2<int&, Y>; // expected-note{{instantiation}}
template struct X2<const void, Y>; // expected-note{{instantiation}}

template<typename T, typename Class, T Class::*Ptr>
struct X3 {
  X3<T, Class, Ptr> &operator=(const T& value) {
    return *this;
  }
};

X3<int, Y, &Y::x> x3;

typedef int Y::*IntMember;

template<IntMember Member>
struct X4 {
  X3<int, Y, Member> member;
  
  int &getMember(Y& y) { return y.*Member; }
};

int &get_X4(X4<&Y::x> x4, Y& y) { 
  return x4.getMember(y); 
}

template<IntMember Member>
void accept_X4(X4<Member>);

void test_accept_X4(X4<&Y::x> x4) {
  accept_X4(x4);
}

namespace ValueDepMemberPointer {
  template <void (*)()> struct instantiate_function {};
  template <typename T> struct S {
    static void instantiate();
    typedef instantiate_function<&S::instantiate> x; // expected-note{{instantiation}}
  };
  template <typename T> void S<T>::instantiate() {
    int a[(int)sizeof(T)-42]; // expected-error{{array with a negative size}}
  }
  S<int> s; 
}

namespace PR18192 {
  struct A { struct { int n; }; };
  template<int A::*> struct X {};
  constexpr int A::*p = &A::n;
  X<p> x; // expected-error{{not a pointer to member constant}}
}
