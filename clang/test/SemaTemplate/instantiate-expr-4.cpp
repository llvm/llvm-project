// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify -std=c++11 %s

// ---------------------------------------------------------------------
// C++ Functional Casts
// ---------------------------------------------------------------------
template<int N>
struct ValueInit0 {
  int f() {
    return int();
  }
};

template struct ValueInit0<5>;

template<int N>
struct FunctionalCast0 {
  int f() {
    return int(N);
  }
};

template struct FunctionalCast0<5>;

struct X { // expected-note 3 {{candidate constructor (the implicit copy constructor)}}
#if __cplusplus >= 201103L
// expected-note@-2 3 {{candidate constructor (the implicit move constructor) not viable}}
#endif
  X(int, int); // expected-note 3 {{candidate constructor}}
};

template<int N, int M>
struct BuildTemporary0 {
  X f() {
    return X(N, M);
  }
};

template struct BuildTemporary0<5, 7>;

template<int N, int M>
struct Temporaries0 {
  void f() {
    (void)X(N, M);
  }
};

template struct Temporaries0<5, 7>;

// Ensure that both the constructor and the destructor are instantiated by
// checking for parse errors from each.
template<int N> struct BadX {
  BadX() { int a[-N]; } // expected-error {{array with a negative size}}
  ~BadX() { int a[-N]; } // expected-error {{array with a negative size}}
};

template<int N>
struct PR6671 {
  void f() { (void)BadX<1>(); } // expected-note 2 {{instantiation}}
};
template struct PR6671<1>;

// ---------------------------------------------------------------------
// new/delete expressions
// ---------------------------------------------------------------------
struct Y { };

template<typename T>
struct New0 {
  T* f(bool x) {
    if (x)
      return new T; // expected-error{{no matching}}
    else
      return new T();
  }
};

template struct New0<int>;
template struct New0<Y>;
template struct New0<X>; // expected-note{{instantiation}}

template<typename T, typename Arg1>
struct New1 {
  T* f(bool x, Arg1 a1) {
    return new T(a1); // expected-error{{no matching}}
  }
};

template struct New1<int, float>;
template struct New1<Y, Y>;
template struct New1<X, Y>; // expected-note{{instantiation}}

template<typename T, typename Arg1, typename Arg2>
struct New2 {
  T* f(bool x, Arg1 a1, Arg2 a2) {
    return new T(a1, a2); // expected-error{{no matching}}
  }
};

template struct New2<X, int, float>;
template struct New2<X, int, int*>; // expected-note{{instantiation}}
// FIXME: template struct New2<int, int, float>;

// PR5833
struct New3 {
  New3();

  void *operator new[](__SIZE_TYPE__) __attribute__((unavailable)); // expected-note{{explicitly made unavailable}}
};

template<class C>
void* object_creator() {
  return new C(); // expected-error{{call to unavailable function 'operator new[]'}}
}

template void *object_creator<New3[4]>(); // expected-note{{instantiation}}

template<typename T>
struct Delete0 {
  void f(T t) {
    delete t; // expected-error{{cannot delete}}
    ::delete [] t; // expected-error{{cannot delete}}
  }
};

template struct Delete0<int*>;
template struct Delete0<X*>;
template struct Delete0<int>; // expected-note{{instantiation}}

namespace PR5755 {
  template <class T>
  void Foo() {
    char* p = 0;
    delete[] p;
  }
  
  void Test() {
    Foo<int>();
  }
}

namespace PR10480 {
  template<typename T>
  struct X {
    X();
    ~X() {
      T *ptr = 1; // expected-error{{cannot initialize a variable of type 'int *' with an rvalue of type 'int'}}
    }
  };

  template<typename T>
  void f() {
    new X<int>[1]; // expected-note{{in instantiation of member function 'PR10480::X<int>::~X' requested here}}
  }

  template void f<int>();
}

// ---------------------------------------------------------------------
// throw expressions
// ---------------------------------------------------------------------
template<typename T>
struct Throw1 {
  void f(T t) {
    throw;
    throw t; // expected-error{{incomplete type}}
  }
};

struct Incomplete; // expected-note 2{{forward}}

template struct Throw1<int>;
template struct Throw1<int*>;
template struct Throw1<Incomplete*>; // expected-note{{instantiation}}

// ---------------------------------------------------------------------
// typeid expressions
// ---------------------------------------------------------------------

namespace std {
  class type_info;
}

template<typename T>
struct TypeId0 {
  const std::type_info &f(T* ptr) {
    if (ptr)
      return typeid(ptr);
    else
      return typeid(T); // expected-error{{'typeid' of incomplete type 'Incomplete'}}
  }
};

struct Abstract {
  virtual void f() = 0;
};

template struct TypeId0<int>;
template struct TypeId0<Incomplete>; // expected-note{{instantiation of member function}}
template struct TypeId0<Abstract>;

// ---------------------------------------------------------------------
// type traits
// ---------------------------------------------------------------------
template<typename T>
struct is_pod {
  static const bool value = __is_pod(T);
};

static int is_pod0[is_pod<X>::value? -1 : 1];
static int is_pod1[is_pod<Y>::value? 1 : -1];

// ---------------------------------------------------------------------
// initializer lists
// ---------------------------------------------------------------------
template<typename T, typename Val1>
struct InitList1 {
  void f(Val1 val1) { 
    T x = { val1 };
#if __cplusplus >= 201103L
    // expected-error@-2 {{type 'float' cannot be narrowed to 'int' in initializer list}}
    // expected-note@-3 {{insert an explicit cast to silence this issue}}
#endif
  }
};

struct APair {
  int *x;
  const float *y;
};

template struct InitList1<int[1], float>;
#if __cplusplus >= 201103L
// expected-note@-2 {{instantiation of member function}}
#endif
template struct InitList1<APair, int*>;

template<typename T, typename Val1, typename Val2>
struct InitList2 {
  void f(Val1 val1, Val2 val2) { 
    T x = { val1, val2 }; // expected-error{{cannot initialize}}
  }
};

template struct InitList2<APair, int*, float*>;
template struct InitList2<APair, int*, double*>; // expected-note{{instantiation}}

// ---------------------------------------------------------------------
// member references
// ---------------------------------------------------------------------
template<typename T, typename Result>
struct DotMemRef0 {
  void f(T t) {
    Result result = t.m; // expected-error{{non-const lvalue reference to type}}
  }
};

struct MemInt {
  int m;
};

struct InheritsMemInt : MemInt { };

struct MemIntFunc {
  static int m(int);
};

template struct DotMemRef0<MemInt, int&>;
template struct DotMemRef0<InheritsMemInt, int&>;
template struct DotMemRef0<MemIntFunc, int (*)(int)>;
template struct DotMemRef0<MemInt, float&>; // expected-note{{instantiation}}

template<typename T, typename Result>
struct ArrowMemRef0 {
  void f(T t) {
    Result result = t->m; // expected-error 2{{non-const lvalue reference}}
  }
};

template<typename T>
struct ArrowWrapper {
  T operator->();
};

template struct ArrowMemRef0<MemInt*, int&>;
template struct ArrowMemRef0<InheritsMemInt*, int&>;
template struct ArrowMemRef0<MemIntFunc*, int (*)(int)>;
template struct ArrowMemRef0<MemInt*, float&>; // expected-note{{instantiation}}

template struct ArrowMemRef0<ArrowWrapper<MemInt*>, int&>;
template struct ArrowMemRef0<ArrowWrapper<InheritsMemInt*>, int&>;
template struct ArrowMemRef0<ArrowWrapper<MemIntFunc*>, int (*)(int)>;
template struct ArrowMemRef0<ArrowWrapper<MemInt*>, float&>; // expected-note{{instantiation}}
template struct ArrowMemRef0<ArrowWrapper<ArrowWrapper<MemInt*> >, int&>;

struct UnresolvedMemRefArray {
  int f(int);
  int f(char);
};
UnresolvedMemRefArray Arr[10];
template<typename U> int UnresolvedMemRefArrayT(U u) {
  return Arr->f(u);
}
template int UnresolvedMemRefArrayT<int>(int);

// FIXME: we should be able to return a MemInt without the reference!
MemInt &createMemInt(int);

template<int N>
struct NonDepMemberExpr0 {
  void f() {
    createMemInt(N).m = N;
  }
};

template struct NonDepMemberExpr0<0>; 

template<typename T, typename Result>
struct MemberFuncCall0 {
  void f(T t) {
    Result result = t.f();
  }
};

template<typename T>
struct HasMemFunc0 {
  T f();
};


template struct MemberFuncCall0<HasMemFunc0<int&>, const int&>;

template<typename Result>
struct ThisMemberFuncCall0 {
  Result g();

  void f() {
    Result r1 = g();
    Result r2 = this->g();
  }
};

template struct ThisMemberFuncCall0<int&>;

template<typename T>
struct NonDepMemberCall0 {
  void foo(HasMemFunc0<int&> x) {
    T result = x.f(); // expected-error{{non-const lvalue reference}}
  }
};

template struct NonDepMemberCall0<int&>;
template struct NonDepMemberCall0<const int&>;
template struct NonDepMemberCall0<float&>; // expected-note{{instantiation}}


template<typename T>
struct QualifiedDeclRef0 {
  T f() {
    return is_pod<X>::value; // expected-error{{non-const lvalue reference to type 'int' cannot bind to a value of unrelated type 'const bool'}}
  }
};

template struct QualifiedDeclRef0<bool>;
template struct QualifiedDeclRef0<int&>; // expected-note{{instantiation}}
