// RUN: %clang_cc1 -fms-compatibility -fsyntax-only -verify %s


template <class T>
class A {
public:
   void f(T a) { }// expected-note {{must qualify identifier to find this declaration in dependent base class}}
   void g();// expected-note {{must qualify identifier to find this declaration in dependent base class}}
};


template <class T>
class B : public A<T> {
public:
	void z(T a)
    {
       f(a); // expected-warning {{use of identifier 'f' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
       g(); // expected-warning {{use of identifier 'g' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
    }
};

template class B<int>; // expected-note {{requested here}}
template class B<char>;

void test()
{
    B<int> b;
    b.z(3);
}

namespace lookup_dependent_bases_id_expr {

template<class T> class A {
public:
  int var;
};


template<class T>
class B : public A<T> {
public:
  void f() {
    var = 3;
  }
};

template class B<int>;

}



namespace lookup_dependent_base_class_static_function {

template <class T>
class A {
public:
   static void static_func();// expected-note {{must qualify identifier to find this declaration in dependent base class}}
   void func();// expected-note {{must qualify identifier to find this declaration in dependent base class}}
};


template <class T>
class B : public A<T> {
public:
  static void z2(){
    static_func();  // expected-warning {{use of identifier 'static_func' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
    func(); // expected-warning {{use of identifier 'func' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}} expected-error {{call to non-static member function without an object argument}}
  }
};
template class B<int>; // expected-note {{requested here}}

} 



namespace lookup_dependent_base_class_default_argument {

template<class T>
class A {
public:
  static int f1(); // expected-note {{must qualify identifier to find this declaration in dependent base class}} 
  int f2(); // expected-note {{must qualify identifier to find this declaration in dependent base class}} 
};

template<class T>
class B : public A<T> {
public:
  void g1(int p = f1());// expected-warning {{use of identifier 'f1' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
  void g2(int p = f2());// expected-warning {{use of identifier 'f2' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}} expected-error {{call to non-static member function without an object argument}}
};

void foo()
{
	B<int> b;
	b.g1(); // expected-note {{required here}}
	b.g2(); // expected-note {{required here}}
}

}


namespace lookup_dependent_base_class_friend {

template <class T>
class B {
public:
  static void g();  // expected-note {{must qualify identifier to find this declaration in dependent base class}} 
};

template <class T>
class A : public B<T> {
public:
  friend void foo(A<T> p){
    g(); // expected-warning {{use of identifier 'g' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
  }
};

int main2()
{
  A<int> a;
  foo(a); // expected-note {{requested here}}
}

}


namespace lookup_dependent_base_no_typo_correction {

class C {
public:
  int m_hWnd;
};

template <class T>
class A : public T {
public:
  void f(int hWnd) {
    m_hWnd = 1;
  }
};

template class A<C>;

}

namespace PR12701 {

class A {};
class B {};

template <class T>
class Base {
 public:
  bool base_fun(void* p) { return false; }  // expected-note {{must qualify identifier to find this declaration in dependent base clas}}
  operator T*() const { return 0; }
};

template <class T>
class Container : public Base<T> {
 public:
  template <typename S>
  bool operator=(const Container<S>& rhs) {
    return base_fun(rhs);  // expected-warning {{use of identifier 'base_fun' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
  }
};

void f() {
  Container<A> text_provider;
  Container<B> text_provider2;
  text_provider2 = text_provider;  // expected-note {{in instantiation of function template specialization}}
}

}  // namespace PR12701
