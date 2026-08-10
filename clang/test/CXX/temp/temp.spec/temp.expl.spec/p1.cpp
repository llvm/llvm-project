// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// This test creates cases where implicit instantiations of various entities
// would cause a diagnostic, but provides expliict specializations for those
// entities that avoid the diagnostic. The intent is to verify that 
// implicit instantiations do not occur (because the explicit specialization 
// is used instead).
struct NonDefaultConstructible {
  NonDefaultConstructible(int);
};


// C++ [temp.expl.spec]p1:
//   An explicit specialization of any of the following:

//     -- function template
template<typename T> void f0(T) {
  T t;
}

template<> void f0(NonDefaultConstructible) { }

void test_f0(NonDefaultConstructible NDC) {
  f0(NDC);
}

//     -- class template
template<typename T>
struct X0 {
  static T member;
  
  void f1(T t) {
    t = 17;
  }
  
  struct Inner : public T { };
  
  template<typename U>
  struct InnerTemplate : public T { };
  
  template<typename U>
  void ft1(T t, U u);
};

template<typename T> 
template<typename U>
void X0<T>::ft1(T t, U u) {
  t = u;
}

template<typename T> T X0<T>::member;

template<> struct X0<void> { };
X0<void> test_X0;
  

//     -- member function of a class template
template<> void X0<void*>::f1(void *) { }

void test_spec(X0<void*> xvp, void *vp) {
  xvp.f1(vp);
}

//     -- static data member of a class template
template<> 
NonDefaultConstructible X0<NonDefaultConstructible>::member = 17;

NonDefaultConstructible &get_static_member() {
  return X0<NonDefaultConstructible>::member;
}

//    -- member class of a class template
template<>
struct X0<void*>::Inner { };

X0<void*>::Inner inner0;

//    -- member class template of a class template
template<>
template<>
struct X0<void*>::InnerTemplate<int> { };

X0<void*>::InnerTemplate<int> inner_template0;

//    -- member function template of a class template
template<>
template<>
void X0<void*>::ft1(void*, const void*) { }

void test_func_template(X0<void *> xvp, void *vp, const void *cvp) {
  xvp.ft1(vp, cvp);
}

// example from the standard:
template<class T> class stream;
template<> class stream<char> { /* ... */ };
template<class T> class Array { /* ... */ }; 
template<class T> void sort(Array<T>& v) { /* ... */ }
template<> void sort<char*>(Array<char*>&) ;
