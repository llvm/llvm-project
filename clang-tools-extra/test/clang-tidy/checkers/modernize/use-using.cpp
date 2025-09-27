// RUN: %check_clang_tidy %s modernize-use-using %t -- -- -fno-delayed-template-parsing -I %S/Inputs/use-using/

typedef long LL;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using LL = long;

typedef int Bla;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Bla = int;

typedef Bla Bla2;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Bla2 = Bla;

typedef const int ConstInt;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using ConstInt = const int;

typedef const int *const ConstPtrToConstInt;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using ConstPtrToConstInt = const int *const;

typedef struct foo_s {} * foo_t;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using foo_t = struct foo_s {} *;

typedef void (*type)(int, int);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using type = void (*)(int, int);

typedef void (*type2)();
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using type2 = void (*)();

class Class {
  typedef long long Type;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'using' instead of 'typedef'
  // CHECK-FIXES: using Type = long long;
};

typedef void (Class::*MyPtrType)(Bla) const;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using MyPtrType = void (Class::*)(Bla)[[ATTR:( __attribute__\(\(thiscall\)\))?]] const;

class Iterable {
public:
  class Iterator {};
};

template <typename T>
class Test {
  typedef typename T::iterator Iter;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'using' instead of 'typedef'
  // CHECK-FIXES: using Iter = typename T::iterator;
};

using balba = long long;

union A {};

typedef void (A::*PtrType)(int, int) const;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using PtrType = void (A::*)(int, int)[[ATTR]] const;

typedef Class some_class;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using some_class = Class;

typedef Class Cclass;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Cclass = Class;

typedef Cclass cclass2;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using cclass2 = Cclass;

class cclass {};

typedef void (cclass::*MyPtrType3)(Bla);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using MyPtrType3 = void (cclass::*)(Bla)[[ATTR]];

using my_class = int;

typedef Test<my_class *> another;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using another = Test<my_class *>;

typedef int* PInt;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using PInt = int*;

typedef int bla1, bla2, bla3;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-MESSAGES: :[[@LINE-2]]:17: warning: use 'using' instead of 'typedef'
// CHECK-MESSAGES: :[[@LINE-3]]:23: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using bla1 = int;
// CHECK-FIXES-NEXT: using bla2 = bla1;
// CHECK-FIXES-NEXT: using bla3 = bla1;

typedef int I, &LVal, &&RVal, *Ptr, *const ConstPtr, Vec3[3], Fn(), (*FnPtr)();
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-MESSAGES: :[[@LINE-2]]:14: warning: use 'using' instead of 'typedef'
// CHECK-MESSAGES: :[[@LINE-3]]:21: warning: use 'using' instead of 'typedef'
// CHECK-MESSAGES: :[[@LINE-4]]:29: warning: use 'using' instead of 'typedef'
// CHECK-MESSAGES: :[[@LINE-5]]:35: warning: use 'using' instead of 'typedef'
// CHECK-MESSAGES: :[[@LINE-6]]:52: warning: use 'using' instead of 'typedef'
// CHECK-MESSAGES: :[[@LINE-7]]:61: warning: use 'using' instead of 'typedef'
// CHECK-MESSAGES: :[[@LINE-8]]:67: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using I = int;
// CHECK-FIXES-NEXT: using LVal = I &;
// CHECK-FIXES-NEXT: using RVal = I &&;
// CHECK-FIXES-NEXT: using Ptr = I *;
// CHECK-FIXES-NEXT: using ConstPtr = I *const;
// CHECK-FIXES-NEXT: using Vec3 = I[3];
// CHECK-FIXES-NEXT: using Fn = I();
// CHECK-FIXES-NEXT: using FnPtr = I (*)();

#define CODE typedef int INT

CODE;
// CHECK-FIXES: #define CODE typedef int INT
// CHECK-FIXES: CODE;

struct Foo;

#define TYPEDEF typedef
TYPEDEF Foo Bak;
// CHECK-FIXES: #define TYPEDEF typedef
// CHECK-FIXES: TYPEDEF Foo Bak;

#define FOO Foo
typedef FOO Bam;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: #define FOO Foo
// CHECK-FIXES: using Bam = FOO;

typedef struct Foo Bap;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Bap = struct Foo;

struct Foo typedef Bap2;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Bap2 =struct Foo ;

Foo typedef Bap3;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Bap3 =Foo ;

typedef struct Unknown Baq;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Baq = struct Unknown;

struct Unknown2 typedef Baw;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Baw =struct Unknown2 ;

int typedef Bax;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Bax =int ;

typedef struct Q1 { int a; } S1;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using S1 = struct Q1 { int a; };
typedef struct { int b; } S2;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using S2 = struct { int b; };
struct Q2 { int c; } typedef S3;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using S3 =struct Q2 { int c; } ;
struct { int d; } typedef S4;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using S4 =struct { int d; } ;

namespace my_space {
  class my_cclass {};
  typedef my_cclass FuncType;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using FuncType = my_cclass;
}

typedef int UnknownExtent[];
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using UnknownExtent = int[];

typedef double Matrix3x3[3][3];
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Matrix3x3 = double[3][3];

#define lol 4
typedef unsigned Map[lol];
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Map = unsigned[lol];

namespace template_instantiations {
template <typename T>
class C {
 protected:
  typedef C<T> super;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'using' instead of 'typedef'
  // CHECK-FIXES: using super = C<T>;
  virtual void f();

public:
  virtual ~C();
};

class D : public C<D> {
  void f() override { super::f(); }
};
class E : public C<E> {
  void f() override { super::f(); }
};
}

template <typename T1, typename T2>
class TwoArgTemplate {
  typedef TwoArgTemplate<T1, T2> self;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'using' instead of 'typedef'
  // CHECK-FIXES: using self = TwoArgTemplate<T1, T2>;
};

template <bool B, typename T>
struct S {};

typedef S<(0 > 0), int> S_t, *S_p;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-MESSAGES: :[[@LINE-2]]:28: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using S_t = S<(0 > 0), int>;
// CHECK-FIXES-NEXT: using S_p = S_t *;

typedef S<(0 < 0), int> S2_t, *S2_p;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-MESSAGES: :[[@LINE-2]]:29: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using S2_t = S<(0 < 0), int>;
// CHECK-FIXES-NEXT: using S2_p = S2_t *;

typedef S<(0 > 0 && (3 > 1) && (1 < 1)), int> S3_t;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using S3_t = S<(0 > 0 && (3 > 1) && (1 < 1)), int>;

template <bool B>
struct Q {};

constexpr bool b[1] = {true};

typedef Q<b[0 < 0]> Q_t, *Q_p;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-MESSAGES: :[[@LINE-2]]:24: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Q_t = Q<b[0 < 0]>;
// CHECK-FIXES-NEXT: using Q_p = Q_t *;

typedef Q<b[0 < 0]> Q2_t;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Q2_t = Q<b[0 < 0]>;

struct T {
  constexpr T(bool) {}

  static constexpr bool b = true;
};

typedef Q<T{0 < 0}.b> Q3_t, *Q3_p;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-MESSAGES: :[[@LINE-2]]:27: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Q3_t = Q<T{0 < 0}.b>;
// CHECK-FIXES-NEXT: using Q3_p = Q3_t *;

typedef Q<T{0 < 0}.b> Q3_t;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Q3_t = Q<T{0 < 0}.b>;

typedef TwoArgTemplate<TwoArgTemplate<int, Q<T{0 < 0}.b> >, S<(0 < 0), Q<b[0 < 0]> > > Nested_t;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Nested_t = TwoArgTemplate<TwoArgTemplate<int, Q<T{0 < 0}.b> >, S<(0 < 0), Q<b[0 < 0]> > >;

template <typename a>
class TemplateKeyword {
  typedef typename a::template b<> d;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'using' instead of 'typedef'
  // CHECK-FIXES: using d = typename a::template b<>;

  typedef typename a::template b<>::c d2;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'using' instead of 'typedef'
  // CHECK-FIXES: using d2 = typename a::template b<>::c;
};

template <typename... Args>
class Variadic {};

typedef Variadic<Variadic<int, bool, Q<T{0 < 0}.b> >, S<(0 < 0), Variadic<Q<b[0 < 0]> > > > Variadic_t;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Variadic_t = Variadic<Variadic<int, bool, Q<T{0 < 0}.b> >, S<(0 < 0), Variadic<Q<b[0 < 0]> > > >;

typedef Variadic<Variadic<int, bool, Q<T{0 < 0}.b> >, S<(0 < 0), Variadic<Q<b[0 < 0]> > > > Variadic_t, *Variadic_p;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-MESSAGES: :[[@LINE-2]]:103: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Variadic_t = Variadic<Variadic<int, bool, Q<T{0 < 0}.b> >, S<(0 < 0), Variadic<Q<b[0 < 0]> > > >;
// CHECK-FIXES-NEXT: using Variadic_p = Variadic_t *;

typedef struct { int a; } R_t, *R_p;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-MESSAGES: :[[@LINE-2]]:30: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using R_t = struct { int a; };
// CHECK-FIXES-NEXT: using R_p = R_t *;

typedef enum { ea1, eb1 } EnumT1;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using EnumT1 = enum { ea1, eb1 };

#include "modernize-use-using.h"

typedef enum { ea2, eb2 } EnumT2_CheckTypedefImpactFromAnotherFile;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using EnumT2_CheckTypedefImpactFromAnotherFile = enum { ea2, eb2 };

template <int A>
struct InjectedClassName {
  typedef InjectedClassName b;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'using' instead of 'typedef'
  // CHECK-FIXES: using b = InjectedClassName;
};

template <int>
struct InjectedClassNameWithUnnamedArgument {
  typedef InjectedClassNameWithUnnamedArgument b;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'using' instead of 'typedef'
  // CHECK-FIXES: using b = InjectedClassNameWithUnnamedArgument;
};

typedef struct { int a; union { int b; }; } PR50990;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using PR50990 = struct { int a; union { int b; }; };

typedef struct { struct { int a; struct { struct { int b; } c; int d; } e; } f; int g; } PR50990_nested;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using PR50990_nested = struct { struct { int a; struct { struct { int b; } c; int d; } e; } f; int g; };

typedef struct { struct { int a; } b; union { int c; float d; struct { int e; }; }; struct { double f; } g; } PR50990_siblings;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using PR50990_siblings = struct { struct { int a; } b; union { int c; float d; struct { int e; }; }; struct { double f; } g; };

typedef void (*ISSUE_65055_1)(int);
typedef bool (*ISSUE_65055_2)(int);
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use 'using' instead of 'typedef'
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using ISSUE_65055_1 = void (*)(int);
// CHECK-FIXES: using ISSUE_65055_2 = bool (*)(int);

typedef class ISSUE_67529_1 *ISSUE_67529;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using ISSUE_67529 = class ISSUE_67529_1 *;

// Some Header
extern "C" {

typedef int InExternC;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef' [modernize-use-using]
// CHECK-FIXES: using InExternC = int;

}

extern "C++" {

typedef int InExternCPP;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef' [modernize-use-using]
// CHECK-FIXES: using InExternCPP = int;

}

namespace ISSUE_72179
{  
  void foo()
  {
    typedef int a;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'using' instead of 'typedef' [modernize-use-using]
    // CHECK-FIXES: using a = int;
  }

  void foo2()
  {
    typedef struct { int a; union { int b; }; } c;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'using' instead of 'typedef' [modernize-use-using]
    // CHECK-FIXES: using c = struct { int a; union { int b; }; };
  }

  template <typename T>
  void foo3()
  {
    typedef T b;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'using' instead of 'typedef' [modernize-use-using]
    // CHECK-FIXES: using b = T;
  }

  template <typename T>
  class MyClass
  {
    void foo()
    {
      typedef MyClass c;
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'using' instead of 'typedef' [modernize-use-using]
      // CHECK-FIXES: using c = MyClass;
    }
  };

  const auto foo4 = [](int a){typedef int d;};
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: use 'using' instead of 'typedef' [modernize-use-using]
  // CHECK-FIXES: const auto foo4 = [](int a){using d = int;};
}

typedef int* IntPtr, *AlsoIntPtr;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef' [modernize-use-using]
// CHECK-MESSAGES: :[[@LINE-2]]:20: warning: use 'using' instead of 'typedef' [modernize-use-using]
// CHECK-FIXES: using IntPtr = int*;
// CHECK-FIXES-NEXT: using AlsoIntPtr = IntPtr *;
// FIXME: the second fix-it is wrong. This is issue #150276.

typedef int (FnWithRedundantParens)();
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef' [modernize-use-using]
// CHECK-FIXES: using FnWithRedundantParens = int ()();
// FIXME: that fix-it makes the code ill-formed. The redundant parentheses need to be removed,
// i.e., it should be 'using FnWithRedundantParens = int ()'. It's not a pressing issue though.

#ifndef SpecialMode
#define SomeMacro(x) x
#else
#define SomeMacro(x) SpecialType
#endif

class SomeMacro(GH33760) { };

typedef void(SomeMacro(GH33760)::* FunctionType)(float, int);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef' [modernize-use-using]
// CHECK-FIXES: using FunctionType = void(SomeMacro(GH33760)::*)(float, int);

#define CDECL __attribute((cdecl))

// GH37846 & GH41685
typedef void (CDECL *GH37846)(int);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef' [modernize-use-using]
// CHECK-FIXES: using GH37846 = void (CDECL *)(int);

typedef void (__attribute((cdecl)) *GH41685)(int);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef' [modernize-use-using]
// CHECK-FIXES: using GH41685 = void (__attribute((cdecl)) *)(int);

namespace GH83568 {
  typedef int(*name)(int arg1, int arg2);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'using' instead of 'typedef' [modernize-use-using]
// CHECK-FIXES: using name = int(*)(int arg1, int arg2);
}

#ifdef FOO
#define GH95716 float
#else
#define GH95716 double
#endif

typedef GH95716 foo;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef' [modernize-use-using]
// CHECK-FIXES: using foo = GH95716;

namespace GH97009 {
  typedef double PointType[3];
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'using' instead of 'typedef' [modernize-use-using]
// CHECK-FIXES: using PointType = double[3];
  typedef bool (*Function)(PointType, PointType);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'using' instead of 'typedef' [modernize-use-using]
// CHECK-FIXES: using Function = bool (*)(PointType, PointType);
}
