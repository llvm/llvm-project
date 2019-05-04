// Header for PCH test cxx-templates.cpp

template <typename T1, typename T2>
struct S;

template <typename T1, typename T2>
struct S {
  S() { }
  static void templ();
};

template <typename T>
struct S<int, T> {
    static void partial();
};

template <>
struct S<int, float> {
    static void explicit_special();
};

template <int x>
int tmpl_f2() { return x; }

template <typename T, int y>
T templ_f(T x) {
  int z = templ_f<int, 5>(3);
  z = tmpl_f2<y+2>();
  T data[y];
  return x+y;
}

void govl(int);
void govl(char);

template <typename T>
struct Unresolv {
  void f() {
    govl(T());
  }
};

template <typename T>
struct Dep {
  typedef typename T::type Ty;
  void f() {
    Ty x = Ty();
    T::my_f();
    int y = T::template my_templf<int>(0);
    ovl(y);
  }
  
  void ovl(int);
  void ovl(float);
};

template<typename T, typename A1>
inline T make_a(const A1& a1) {
  T::depend_declref();
  return T(a1);
}

template <class T> class UseBase {
  void foo();
  typedef int bar;
};

template <class T> class UseA : public UseBase<T> {
  using UseBase<T>::foo;
  using typename UseBase<T>::bar; 
};

template <class T> class Sub : public UseBase<int> { };

template <class _Ret, class _Tp>
  class mem_fun_t
  {
  public:
    explicit
    mem_fun_t(_Ret (_Tp::*__pf)())
     {}

  private:
    _Ret (_Tp::*_M_f)();
  };

template<unsigned N>
bool isInt(int x);

template<> bool isInt<8>(int x) {
  try { ++x; } catch(...) { --x; }
  return true;
}

template<typename _CharT>
int __copy_streambufs_eof(_CharT);

class basic_streambuf 
{
  void m() { }
  friend int __copy_streambufs_eof<>(int);
};

// PR 7660
template<typename T> struct S_PR7660 { void g(void (*)(T)); };
 template<> void S_PR7660<int>::g(void(*)(int)) {}

// PR 7670
template<typename> class C_PR7670;
template<> class C_PR7670<int>;
template<> class C_PR7670<int>;

template <bool B>
struct S2 {
    static bool V;
};

extern template class S2<true>;

template <typename T>
struct S3 {
    void m();
};

template <typename T>
inline void S3<T>::m() { }

template <typename T>
struct S4 {
    void m() { }
};
extern template struct S4<int>;

void S4ImplicitInst() {
    S4<int> s;
    s.m();
}

struct S5 {
  S5(int x);
};

struct TS5 {
  S5 s;
  template <typename T>
  TS5(T y) : s(y) {}
};

// PR 8134
template<class T> void f_PR8134(T);
template<class T> void f_PR8134(T);
void g_PR8134() { f_PR8134(0); f_PR8134('x'); }

// rdar8580149
template <typename T>
struct S6;

template <typename T, unsigned N>
struct S6<const T [N]>
{
private:
   typedef const T t1[N];
public:
   typedef t1& t2;
};

template<typename T>
  struct S7;

template<unsigned N>
struct S7<int[N]> : S6<const int[N]> { };

// Zero-length template argument lists
namespace ZeroLengthExplicitTemplateArgs {
  template<typename T> void h();

  struct Y { 
    template<typename T> void f();
  };

  template<typename T>
    void f(T *ptr) {
    T::template g<>(17);
    ptr->template g2<>(17);
    h<T>();
    h<int>();
    Y y;
    y.f<int>();
  }

  struct X {
    template<typename T> static void g(T);
    template<typename T> void g2(T);
  };
}

namespace NonTypeTemplateParmContext {
  template<typename T, int inlineCapacity = 0> class Vector { };

  struct String {
    template<int inlineCapacity>
    static String adopt(Vector<char, inlineCapacity>&);
  };

  template<int inlineCapacity>
    inline bool equalIgnoringNullity(const Vector<char, inlineCapacity>& a, const String& b) { return false; }
}

// <rdar://problem/11112464>
template< typename > class Foo;

template< typename T >
class Foo : protected T
{
 public:
  Foo& operator=( const Foo& other );
};

template<typename...A> struct NestedExpansion {
  template<typename...B> auto f(A...a, B...b) -> decltype(g(a + b...));
};
template struct NestedExpansion<char, char, char>;

namespace rdar13135282 {
template < typename _Alloc >
void foo(_Alloc = _Alloc());

template < bool > class __pool;

template < template < bool > class _PoolTp >
struct __common_pool {
  typedef _PoolTp < 0 > pool_type;
};

template < template < bool > class _PoolTp >
struct __common_pool_base : __common_pool < _PoolTp > {};

template < template < bool > class _PoolTp >
struct A : __common_pool_base < _PoolTp > {};

template < typename _Poolp = A < __pool > >
struct __mt_alloc {
  typedef typename _Poolp::pool_type __pool_type;
  __mt_alloc() {
    foo<__mt_alloc<> >();
  }
};
}

namespace PR13020 {
template<typename T>
void f() {
 enum E {
   enumerator
 };

 T t = enumerator;
}

template void f<int>();
}

template<typename T> void doNotDeserialize() {}
template<typename T> struct ContainsDoNotDeserialize {
  static int doNotDeserialize;
};
template<typename T> struct ContainsDoNotDeserialize2 {
  static void doNotDeserialize();
};
template<typename T> int ContainsDoNotDeserialize<T>::doNotDeserialize = 0;
template<typename T> void ContainsDoNotDeserialize2<T>::doNotDeserialize() {}


template<typename T> void DependentSpecializedFunc(T x) { x.foo(); }
template<typename T> class DependentSpecializedFuncClass {
  void foo() {}
  friend void DependentSpecializedFunc<>(DependentSpecializedFuncClass);
};

namespace cyclic_module_load {
  // Reduced from a libc++ modules crasher.
  namespace std {
    template<class> class mask_array;
    template<class> class valarray {
    public:
      valarray(const valarray &v);
    };

    class gslice {
      valarray<int> x;
      valarray<int> stride() const { return x; }
    };

    template<class> class mask_array {
      template<class> friend class valarray;
    };
  }
}

namespace local_extern {
  template<typename T> int f() {
    extern int arr[3];
    {
      extern T arr;
      return sizeof(arr);
    }
  }
  template<typename T> int g() {
    extern int arr[3];
    extern T arr;
    return sizeof(arr);
  }
}

namespace rdar15468709a {
  template<typename> struct decay {};

  template<typename FooParamTy> auto foo(FooParamTy fooParam) -> decltype(fooParam);
  template<typename BarParamTy> auto bar(BarParamTy barParam) -> decay<decltype(barParam)>;

  struct B {};

  void crash() {
    B some;
    bar(some);
  }
}

namespace rdar15468709b {
  template<typename> struct decay {};

  template<typename... Foos> int returnsInt(Foos... foos);

  template<typename... FooParamTy> auto foo(FooParamTy... fooParam) -> decltype(returnsInt(fooParam...));
  template<typename... BarParamTy> auto bar(BarParamTy... barParam) -> decay<decltype(returnsInt(barParam...))>;

  struct B {};

  void crash() {
    B some;
    bar(some);
  }
}

namespace rdar15468709c {
  template<typename> struct decay {};

  template<class... Foos> int returnsInt(Foos... foos);

  template<typename FooParamTy> void foo(FooParamTy fooParam) { decltype(fooParam) a; }
  template<typename BarParamTy> auto bar(BarParamTy barParam) -> decay<decltype(barParam)>;

  struct B {};

  void crash() {
    B some;
    bar(some);
  }
}

namespace MemberSpecializationLocation {
  template<typename T> struct A { static int n; };
}

// https://bugs.llvm.org/show_bug.cgi?id=34728
namespace PR34728 {

// case 1: defaulted `NonTypeTemplateParmDecl`, non-defaulted 2nd tpl param
template <int foo = 10, class T>
int func1(T const &);

template <int foo, class T>
int func1(T const &) {
  return foo;
}

// case 2: defaulted `TemplateTypeParmDecl`, non-defaulted 2nd tpl param
template <class A = int, class B>
A func2(B const &);

template <class A, class B>
A func2(B const &) {
  return A(20.0f);
}

// case 3: defaulted `TemplateTemplateParmDecl`, non-defaulted 2nd tpl param
template <class T>
struct Container { T const &item; };

template <template <class> class C = Container, class D>
C<D> func3(D const &);

template <template <class> class C, class D>
C<D> func3(D const &d) {
  return Container<D>{d};
}

} // end namespace PR34728

namespace ClassScopeExplicitSpecializations {
  template<int> struct A {
    template<int> constexpr int f() const { return 1; }
    template<> constexpr int f<0>() const { return 2; }
  };

  template<> template<int> constexpr int A<0>::f() const { return 3; }
  template<> template<> constexpr int A<0>::f<0>() const { return 4; }
  template<> template<> constexpr int A<0>::f<1>() const { return 5; }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winstantiation-after-specialization"
  template int A<2>::f<0>() const;
#pragma clang diagnostic pop
  template int A<2>::f<1>() const;
  extern template int A<3>::f<0>() const;
  extern template int A<3>::f<1>() const;

  template<int> struct B {
    template<typename> static const int v = 1;
    template<typename T> static const int v<T*> = 2;
    template<> static const int v<int> = 3;

    template<typename> static constexpr int w = 1;
    template<typename T> static constexpr int w<T*> = 2;
    template<> static constexpr int w<int> = 3;
  };

  template<> template<typename> constexpr int B<0>::v = 4;
  template<> template<typename T> constexpr int B<0>::v<T*> = 5;
  template<> template<typename T> constexpr int B<0>::v<T&> = 6;
  // This is ill-formed: the initializer of v<int> is instantiated with the
  // class.
  //template<> template<> constexpr int B<0>::v<int> = 7;
  template<> template<> constexpr int B<0>::v<float> = 8;

  template<> template<typename> constexpr int B<0>::w = 4;
  template<> template<typename T> constexpr int B<0>::w<T*> = 5;
  template<> template<typename T> constexpr int B<0>::w<T&> = 6;
  template<> template<> constexpr int B<0>::w<int> = 7;
  template<> template<> constexpr int B<0>::w<float> = 8;
}
