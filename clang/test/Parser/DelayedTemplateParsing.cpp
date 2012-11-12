// RUN: %clang_cc1 -fms-extensions -fdelayed-template-parsing -fsyntax-only -verify -std=c++11 %s

template <class T>
class A {
   void foo() {
       undeclared();
   }
   void foo2();
};

template <class T>
class B {
   void foo4() { } // expected-note {{previous definition is here}}  expected-note {{previous definition is here}}
   void foo4() { } // expected-error {{class member cannot be redeclared}} expected-error {{redefinition of 'foo4'}}  expected-note {{previous definition is here}}

   friend void foo3() {
       undeclared();
   }
};


template <class T>
void B<T>::foo4() {// expected-error {{redefinition of 'foo4'}}
}

template <class T>
void A<T>::foo2() {
    undeclared();
}


template <class T>
void foo3() {
   undeclared();
}

template void A<int>::foo2();


void undeclared()
{

}

template <class T> void foo5() {} //expected-note {{previous definition is here}} 
template <class T> void foo5() {} // expected-error {{redefinition of 'foo5'}}

              

namespace Inner_Outer_same_template_param_name {              

template <class T>
class Outmost {
public:
    template <class T>
    class Inner {
    public:
        void f() {
            T* var;
        }
   };
};

}


namespace PR11931 {

template <typename RunType>
struct BindState;

  template<>
struct BindState<void(void*)> {
  static void Run() { }
};

class Callback {
public:
  typedef void RunType();

  template <typename RunType>
  Callback(BindState<RunType> bind_state) {
    BindState<RunType>::Run();
  }
};


Callback Bind() {
  return Callback(BindState<void(void*)>());
}

}

namespace rdar11700604 {
  template<typename T> void foo() = delete;

  struct X {
    X() = default;

    template<typename T> void foo() = delete;
  };
}

