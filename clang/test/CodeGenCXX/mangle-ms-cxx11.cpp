// RUN: %clang_cc1 -std=c++11 -fms-extensions -emit-llvm %s -o - -triple=i386-pc-win32 -fms-compatibility-version=19.00 | FileCheck %s --check-prefix=CHECK --check-prefix=MSVC2015
// RUN: %clang_cc1 -std=c++11 -fms-extensions -emit-llvm %s -o - -triple=i386-pc-win32 -fms-compatibility-version=18.00 | FileCheck %s --check-prefix=CHECK --check-prefix=MSVC2013

namespace FTypeWithQuals {
template <typename T>
struct S {};

using A = int () const;
S<A> a;
// CHECK-DAG: @"\01?a@FTypeWithQuals@@3U?$S@$$A8@@BAHXZ@1@A"

using B = int () volatile;
S<B> b;
// CHECK-DAG: @"\01?b@FTypeWithQuals@@3U?$S@$$A8@@CAHXZ@1@A"

using C = int () __restrict;
S<C> c;
// CHECK-DAG: @"\01?c@FTypeWithQuals@@3U?$S@$$A8@@IAAHXZ@1@A"

using D = int () const &;
S<D> d;
// CHECK-DAG: @"\01?d@FTypeWithQuals@@3U?$S@$$A8@@GBAHXZ@1@A"

using E = int () volatile &;
S<E> e;
// CHECK-DAG: @"\01?e@FTypeWithQuals@@3U?$S@$$A8@@GCAHXZ@1@A"

using F = int () __restrict &;
S<F> f;
// CHECK-DAG: @"\01?f@FTypeWithQuals@@3U?$S@$$A8@@IGAAHXZ@1@A"

using G = int () const &&;
S<G> g;
// CHECK-DAG: @"\01?g@FTypeWithQuals@@3U?$S@$$A8@@HBAHXZ@1@A"

using H = int () volatile &&;
S<H> h;
// CHECK-DAG: @"\01?h@FTypeWithQuals@@3U?$S@$$A8@@HCAHXZ@1@A"

using I = int () __restrict &&;
S<I> i;
// CHECK-DAG: @"\01?i@FTypeWithQuals@@3U?$S@$$A8@@IHAAHXZ@1@A"

using J = int ();
S<J> j;
// CHECK-DAG: @"\01?j@FTypeWithQuals@@3U?$S@$$A6AHXZ@1@A"

using K = int () &;
S<K> k;
// CHECK-DAG: @"\01?k@FTypeWithQuals@@3U?$S@$$A8@@GAAHXZ@1@A"

using L = int () &&;
S<L> l;
// CHECK-DAG: @"\01?l@FTypeWithQuals@@3U?$S@$$A8@@HAAHXZ@1@A"
}

// CHECK: "\01?DeducedType@@3HA"
auto DeducedType = 30;

// CHECK-DAG: @"\01?Char16Var@@3_SA"
char16_t Char16Var;

// CHECK-DAG: @"\01?Char32Var@@3_UA"
char32_t Char32Var;

// CHECK: "\01?LRef@@YAXAAH@Z"
void LRef(int& a) { }

// CHECK: "\01?RRef@@YAH$$QAH@Z"
int RRef(int&& a) { return a; }

// CHECK: "\01?Null@@YAX$$T@Z"
namespace std { typedef decltype(__nullptr) nullptr_t; }
void Null(std::nullptr_t) {}

namespace EnumMangling {
  extern enum Enum01 { } Enum;
  extern enum Enum02 : bool { } BoolEnum;
  extern enum Enum03 : char { } CharEnum;
  extern enum Enum04 : signed char { } SCharEnum;
  extern enum Enum05 : unsigned char { } UCharEnum;
  extern enum Enum06 : short { } SShortEnum;
  extern enum Enum07 : unsigned short { } UShortEnum;
  extern enum Enum08 : int { } SIntEnum;
  extern enum Enum09 : unsigned int { } UIntEnum;
  extern enum Enum10 : long { } SLongEnum;
  extern enum Enum11 : unsigned long { } ULongEnum;
  extern enum Enum12 : long long { } SLongLongEnum;
  extern enum Enum13 : unsigned long long { } ULongLongEnum;
// CHECK-DAG: @"\01?Enum@EnumMangling@@3W4Enum01@1@A"
// CHECK-DAG: @"\01?BoolEnum@EnumMangling@@3W4Enum02@1@A
// CHECK-DAG: @"\01?CharEnum@EnumMangling@@3W4Enum03@1@A
// CHECK-DAG: @"\01?SCharEnum@EnumMangling@@3W4Enum04@1@A
// CHECK-DAG: @"\01?UCharEnum@EnumMangling@@3W4Enum05@1@A
// CHECK-DAG: @"\01?SShortEnum@EnumMangling@@3W4Enum06@1@A"
// CHECK-DAG: @"\01?UShortEnum@EnumMangling@@3W4Enum07@1@A"
// CHECK-DAG: @"\01?SIntEnum@EnumMangling@@3W4Enum08@1@A"
// CHECK-DAG: @"\01?UIntEnum@EnumMangling@@3W4Enum09@1@A"
// CHECK-DAG: @"\01?SLongEnum@EnumMangling@@3W4Enum10@1@A"
// CHECK-DAG: @"\01?ULongEnum@EnumMangling@@3W4Enum11@1@A"
// CHECK-DAG: @"\01?SLongLongEnum@EnumMangling@@3W4Enum12@1@A"
// CHECK-DAG: @"\01?ULongLongEnum@EnumMangling@@3W4Enum13@1@A"
  decltype(Enum) *UseEnum() { return &Enum; }
  decltype(BoolEnum) *UseBoolEnum() { return &BoolEnum; }
  decltype(CharEnum) *UseCharEnum() { return &CharEnum; }
  decltype(SCharEnum) *UseSCharEnum() { return &SCharEnum; }
  decltype(UCharEnum) *UseUCharEnum() { return &UCharEnum; }
  decltype(SShortEnum) *UseSShortEnum() { return &SShortEnum; }
  decltype(UShortEnum) *UseUShortEnum() { return &UShortEnum; }
  decltype(SIntEnum) *UseSIntEnum() { return &SIntEnum; }
  decltype(UIntEnum) *UseUIntEnum() { return &UIntEnum; }
  decltype(SLongEnum) *UseSLongEnum() { return &SLongEnum; }
  decltype(ULongEnum) *UseULongEnum() { return &ULongEnum; }
  decltype(SLongLongEnum) *UseSLongLongEnum() { return &SLongLongEnum; }
  decltype(ULongLongEnum) *UseULongLongEnum() { return &ULongLongEnum; }
  extern enum class EnumClass01 { } EnumClass;
  extern enum class EnumClass02 : bool { } BoolEnumClass;
  extern enum class EnumClass03 : char { } CharEnumClass;
  extern enum class EnumClass04 : signed char { } SCharEnumClass;
  extern enum class EnumClass05 : unsigned char { } UCharEnumClass;
  extern enum class EnumClass06 : short { } SShortEnumClass;
  extern enum class EnumClass07 : unsigned short { } UShortEnumClass;
  extern enum class EnumClass08 : int { } SIntEnumClass;
  extern enum class EnumClass09 : unsigned int { } UIntEnumClass;
  extern enum class EnumClass10 : long { } SLongEnumClass;
  extern enum class EnumClass11 : unsigned long { } ULongEnumClass;
  extern enum class EnumClass12 : long long { } SLongLongEnumClass;
  extern enum class EnumClass13 : unsigned long long { } ULongLongEnumClass;
// CHECK-DAG: @"\01?EnumClass@EnumMangling@@3W4EnumClass01@1@A"
// CHECK-DAG: @"\01?BoolEnumClass@EnumMangling@@3W4EnumClass02@1@A
// CHECK-DAG: @"\01?CharEnumClass@EnumMangling@@3W4EnumClass03@1@A
// CHECK-DAG: @"\01?SCharEnumClass@EnumMangling@@3W4EnumClass04@1@A
// CHECK-DAG: @"\01?UCharEnumClass@EnumMangling@@3W4EnumClass05@1@A
// CHECK-DAG: @"\01?SShortEnumClass@EnumMangling@@3W4EnumClass06@1@A"
// CHECK-DAG: @"\01?UShortEnumClass@EnumMangling@@3W4EnumClass07@1@A"
// CHECK-DAG: @"\01?SIntEnumClass@EnumMangling@@3W4EnumClass08@1@A"
// CHECK-DAG: @"\01?UIntEnumClass@EnumMangling@@3W4EnumClass09@1@A"
// CHECK-DAG: @"\01?SLongEnumClass@EnumMangling@@3W4EnumClass10@1@A"
// CHECK-DAG: @"\01?ULongEnumClass@EnumMangling@@3W4EnumClass11@1@A"
// CHECK-DAG: @"\01?SLongLongEnumClass@EnumMangling@@3W4EnumClass12@1@A"
// CHECK-DAG: @"\01?ULongLongEnumClass@EnumMangling@@3W4EnumClass13@1@A"
  decltype(EnumClass) *UseEnumClass() { return &EnumClass; }
  decltype(BoolEnumClass) *UseBoolEnumClass() { return &BoolEnumClass; }
  decltype(CharEnumClass) *UseCharEnumClass() { return &CharEnumClass; }
  decltype(SCharEnumClass) *UseSCharEnumClass() { return &SCharEnumClass; }
  decltype(UCharEnumClass) *UseUCharEnumClass() { return &UCharEnumClass; }
  decltype(SShortEnumClass) *UseSShortEnumClass() { return &SShortEnumClass; }
  decltype(UShortEnumClass) *UseUShortEnumClass() { return &UShortEnumClass; }
  decltype(SIntEnumClass) *UseSIntEnumClass() { return &SIntEnumClass; }
  decltype(UIntEnumClass) *UseUIntEnumClass() { return &UIntEnumClass; }
  decltype(SLongEnumClass) *UseSLongEnumClass() { return &SLongEnumClass; }
  decltype(ULongEnumClass) *UseULongEnumClass() { return &ULongEnumClass; }
  decltype(SLongLongEnumClass) *UseSLongLongEnumClass() { return &SLongLongEnumClass; }
  decltype(ULongLongEnumClass) *UseULongLongEnumClass() { return &ULongLongEnumClass; }
}

namespace PR18022 {

struct { } a;
decltype(a) fun(decltype(a) x, decltype(a)) { return x; }
// CHECK-DAG: @"\01?fun@PR18022@@YA?AU<unnamed-type-a>@1@U21@0@Z"

void use_fun() { fun(a, a); }

}

inline int define_lambda() {
  static auto lambda = [] { static int local; ++local; return local; };
// First, we have the static local variable of type "<lambda_1>" inside of
// "define_lambda".
// CHECK-DAG: @"\01?lambda@?1??define_lambda@@YAHXZ@4V<lambda_1>@?0??1@YAHXZ@A"
// Next, we have the "operator()" for "<lambda_1>" which is inside of
// "define_lambda".
// CHECK-DAG: @"\01??R<lambda_1>@?0??define_lambda@@YAHXZ@QBE@XZ"
// Finally, we have the local which is inside of "<lambda_1>" which is inside of
// "define_lambda". Hooray.
// MSVC2013-DAG: @"\01?local@?2???R<lambda_1>@?0??define_lambda@@YAHXZ@QBE@XZ@4HA"
// MSVC2015-DAG: @"\01?local@?1???R<lambda_1>@?0??define_lambda@@YAHXZ@QBE@XZ@4HA"
  return lambda();
}

template <typename T>
void use_lambda_arg(T) {}

inline void call_with_lambda_arg1() {
  use_lambda_arg([]{});
  // CHECK-DAG: @"\01??$use_lambda_arg@V<lambda_1>@?0??call_with_lambda_arg1@@YAXXZ@@@YAXV<lambda_1>@?0??call_with_lambda_arg1@@YAXXZ@@Z"
}

inline void call_with_lambda_arg2() {
  use_lambda_arg([]{});
  // CHECK-DAG: @"\01??$use_lambda_arg@V<lambda_1>@?0??call_with_lambda_arg2@@YAXXZ@@@YAXV<lambda_1>@?0??call_with_lambda_arg2@@YAXXZ@@Z"
}

int call_lambda() {
  call_with_lambda_arg1();
  call_with_lambda_arg2();
  return define_lambda();
}

namespace PR19361 {
struct A {
  void foo() __restrict &;
  void foo() __restrict &&;
};
void A::foo() __restrict & {}
// CHECK-DAG: @"\01?foo@A@PR19361@@QIGAEXXZ"
void A::foo() __restrict && {}
// CHECK-DAG: @"\01?foo@A@PR19361@@QIHAEXXZ"
}

int operator"" _deg(long double) { return 0; }
// CHECK-DAG: @"\01??__K_deg@@YAHO@Z"

template <char...>
void templ_fun_with_pack() {}

template void templ_fun_with_pack<>();
// CHECK-DAG: @"\01??$templ_fun_with_pack@$S@@YAXXZ"

template <typename...>
void templ_fun_with_ty_pack() {}

template void templ_fun_with_ty_pack<>();
// MSVC2013-DAG: @"\01??$templ_fun_with_ty_pack@$$$V@@YAXXZ"
// MSVC2015-DAG: @"\01??$templ_fun_with_ty_pack@$$V@@YAXXZ"

template <template <class> class...>
void templ_fun_with_templ_templ_pack() {}

template void templ_fun_with_templ_templ_pack<>();
// MSVC2013-DAG: @"\01??$templ_fun_with_templ_templ_pack@$$$V@@YAXXZ"
// MSVC2015-DAG: @"\01??$templ_fun_with_templ_templ_pack@$$V@@YAXXZ"

namespace PR20047 {
template <typename T>
struct A {};

template <typename T>
using AliasA = A<T>;

template <template <typename> class>
void f() {}

template void f<AliasA>();
// CHECK-DAG: @"\01??$f@$$YAliasA@PR20047@@@PR20047@@YAXXZ"
}

namespace UnnamedType {
struct A {
  struct {} *TD;
};

void f(decltype(*A::TD)) {}
// CHECK-DAG: @"\01?f@UnnamedType@@YAXAAU<unnamed-type-TD>@A@1@@Z"

template <typename T>
struct B {
  enum {
  } *e;
};

void f(decltype(B<int>::e)) {}
// CHECK-DAG: @"\01?f@UnnamedType@@YAXPAW4<unnamed-type-e>@?$B@H@1@@Z
}

namespace PR24651 {
template <typename T>
void f(T) {}

void g() {
  enum {} E;
  f(E);
  {
    enum {} E;
    f(E);
  }
}
// CHECK-DAG: @"\01??$f@W4<unnamed-type-E>@?1??g@PR24651@@YAXXZ@@PR24651@@YAXW4<unnamed-type-E>@?1??g@0@YAXXZ@@Z"
// CHECK-DAG: @"\01??$f@W4<unnamed-type-E>@?2??g@PR24651@@YAXXZ@@PR24651@@YAXW4<unnamed-type-E>@?2??g@0@YAXXZ@@Z"
}

namespace PR18204 {
template <typename T>
int f(T *) { return 0; }
static union {
  int n = f(this);
};
// CHECK-DAG: @"\01??$f@T<unnamed-type-$S1>@PR18204@@@PR18204@@YAHPAT<unnamed-type-$S1>@0@@Z"
}

int PR26105() {
  auto add = [](int x) { return ([x](int y) { return x + y; }); };
  return add(3)(4);
}
// CHECK-DAG: @"\01??R<lambda_0>@?0??PR26105@@YAHXZ@QBE@H@Z"
// CHECK-DAG: @"\01??R<lambda_1>@?0???R<lambda_0>@?0??PR26105@@YAHXZ@QBE@H@Z@QBE@H@Z"

int __unaligned * unaligned_foo1() { return 0; }
int __unaligned * __unaligned * unaligned_foo2() { return 0; }
__unaligned int unaligned_foo3() { return 0; }
void unaligned_foo4(int __unaligned *p1) {}
void unaligned_foo5(int __unaligned * __restrict p1) {}
template <typename T> T unaligned_foo6(T t) { return t; }
void unaligned_foo7() { unaligned_foo6<int *>(0); unaligned_foo6<int __unaligned *>(0); }

// CHECK-DAG: @"\01?unaligned_foo1@@YAPFAHXZ"
// CHECK-DAG: @"\01?unaligned_foo2@@YAPFAPFAHXZ"
// CHECK-DAG: @"\01?unaligned_foo3@@YAHXZ"
// CHECK-DAG: @"\01?unaligned_foo4@@YAXPFAH@Z"
// CHECK-DAG: @"\01?unaligned_foo5@@YAXPIFAH@Z"
// CHECK-DAG: @"\01??$unaligned_foo6@PAH@@YAPAHPAH@Z"
// CHECK-DAG: @"\01??$unaligned_foo6@PFAH@@YAPFAHPFAH@Z"

// __unaligned qualifier for function types
struct unaligned_foo8_S {
    void unaligned_foo8() volatile __unaligned;
};
void unaligned_foo8_S::unaligned_foo8() volatile __unaligned {}

// CHECK-DAG: @"\01?unaligned_foo8@unaligned_foo8_S@@QFCEXXZ"

namespace PR31197 {
struct A {
  // CHECK-DAG: define linkonce_odr x86_thiscallcc i32* @"\01??R<lambda_1>@x@A@PR31197@@QBE@XZ"(
  int *x = []() {
    static int white;
    // CHECK-DAG: @"\01?white@?1???R<lambda_1>@x@A@PR31197@@QBE@XZ@4HA"
    return &white;
  }();
  // CHECK-DAG: define linkonce_odr x86_thiscallcc i32* @"\01??R<lambda_1>@y@A@PR31197@@QBE@XZ"(
  int *y = []() {
    static int black;
    // CHECK-DAG: @"\01?black@?1???R<lambda_1>@y@A@PR31197@@QBE@XZ@4HA"
    return &black;
  }();
  using FPtrTy = void(void);
  static void default_args(FPtrTy x = [] {}, FPtrTy y = [] {}, int z = [] { return 1; }() + [] { return 2; }()) {}
  // CHECK-DAG: @"\01??R<lambda_1_1>@?0??default_args@A@PR31197@@SAXP6AXXZ0H@Z@QBE@XZ"(
  // CHECK-DAG: @"\01??R<lambda_1_2>@?0??default_args@A@PR31197@@SAXP6AXXZ0H@Z@QBE@XZ"(
  // CHECK-DAG: @"\01??R<lambda_2_1>@?0??default_args@A@PR31197@@SAXP6AXXZ0H@Z@QBE@XZ"(
  // CHECK-DAG: @"\01??R<lambda_3_1>@?0??default_args@A@PR31197@@SAXP6AXXZ0H@Z@QBE@XZ"(
};
A a;

int call_it = (A::default_args(), 1);
}

enum { enumerator };
void f(decltype(enumerator)) {}
// CHECK-DAG: define internal void @"\01?f@@YAXW4<unnamed-enum-enumerator>@@@Z"(
void use_f() { f(enumerator); }
