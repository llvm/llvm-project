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
   void foo4() { } // expected-note {{previous definition is here}}
   void foo4() { } // expected-error {{class member cannot be redeclared}}
   void foo5() { } // expected-note {{previous definition is here}}

   friend void foo3() {
       undeclared();
   }
};


template <class T>
void B<T>::foo5() { // expected-error {{redefinition of 'foo5'}}
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

namespace PR17334 {

template <typename = void> struct ArrayRef {
  constexpr ArrayRef() {}
};
template <typename = void> void CreateConstInBoundsGEP2_32() {
  ArrayRef<> IdxList;
}
void LLVMBuildStructGEP() { CreateConstInBoundsGEP2_32(); }

}

namespace PR17661 {
template <typename T>
constexpr T Fun(T A) { return T(0); }

constexpr int Var = Fun(20);
}

template <typename T>
auto invalidTrailingRetType() -> Bogus {} // expected-error {{unknown type name 'Bogus'}}

namespace PR19613 {

struct HeapTypeConfig {
  static void from_bitset();
};

template <class Config>
struct TypeImpl  {
  struct BitsetType;

  static void Any() {
    BitsetType::New();
  }
};

template<class Config>
struct TypeImpl<Config>::BitsetType {
  static void New() {
    Config::from_bitset();
  }
};

static void f() {
  TypeImpl<HeapTypeConfig>::Any();
}

template<typename A> struct S {
  template<typename B> struct T;
};
template<typename A> template<typename B> struct S<A>::T {
  template<typename C, typename D> struct U;
  template<typename C> struct U<C, C> {
    template<typename E> static int f() {
      return sizeof(A) + sizeof(B) + sizeof(C) + sizeof(E);
    }
  };
};

static void g() {
  S<int>::T<int>::U<int,int>::f<int>();
}

template<typename T> struct SS {
  template<typename U> struct X;
  template<typename U> struct X<U*>;
};
template<typename T> template<typename U> struct SS<T>::X<U*> {
  static int f() {
    return sizeof(T) + sizeof(U);
  }
};

static void h() {
  SS<int>::X<int*>::f();
}

}

struct PR38460 {
  template <typename>
  struct T {
    static void foo() {
      struct U {
        void dummy() {
          use_delayed_identifier();
        }
      };
    }
  };
};
void use_delayed_identifier();
void trigger_PR38460() {
  PR38460::T<int>::foo();
}

template <typename> struct PR38460_2 {
  struct p {
    struct G {
      bool operator()(int) {}
    };
  };
  static void as() {
    typename p::G g;
    g(0);
  }
};
template struct PR38460_2<int>;
