// RUN: %clang_cc1 -std=c++11 -fms-extensions -emit-llvm %s -o - -triple=i386-pc-win32 -fms-compatibility-version=19.00 | FileCheck %s --check-prefix=CHECK --check-prefix=MSVC2015
// RUN: %clang_cc1 -std=c++11 -fms-extensions -emit-llvm %s -o - -triple=i386-pc-win32 -fms-compatibility-version=18.00 | FileCheck %s --check-prefix=CHECK --check-prefix=MSVC2013
// RUN: %clang_cc1 -std=c++11 -fms-extensions -emit-llvm %s -o - -triple=i386-pc-win32 -gcodeview -debug-info-kind=limited | FileCheck %s --check-prefix=DBG

namespace FTypeWithQuals {
template <typename T>
struct S {};

using A = int () const;
S<A> a;
// CHECK-DAG: @"?a@FTypeWithQuals@@3U?$S@$$A8@@BAHXZ@1@A"

using B = int () volatile;
S<B> b;
// CHECK-DAG: @"?b@FTypeWithQuals@@3U?$S@$$A8@@CAHXZ@1@A"

using C = int () __restrict;
S<C> c;
// CHECK-DAG: @"?c@FTypeWithQuals@@3U?$S@$$A8@@IAAHXZ@1@A"

using D = int () const &;
S<D> d;
// CHECK-DAG: @"?d@FTypeWithQuals@@3U?$S@$$A8@@GBAHXZ@1@A"

using E = int () volatile &;
S<E> e;
// CHECK-DAG: @"?e@FTypeWithQuals@@3U?$S@$$A8@@GCAHXZ@1@A"

using F = int () __restrict &;
S<F> f;
// CHECK-DAG: @"?f@FTypeWithQuals@@3U?$S@$$A8@@IGAAHXZ@1@A"

using G = int () const &&;
S<G> g;
// CHECK-DAG: @"?g@FTypeWithQuals@@3U?$S@$$A8@@HBAHXZ@1@A"

using H = int () volatile &&;
S<H> h;
// CHECK-DAG: @"?h@FTypeWithQuals@@3U?$S@$$A8@@HCAHXZ@1@A"

using I = int () __restrict &&;
S<I> i;
// CHECK-DAG: @"?i@FTypeWithQuals@@3U?$S@$$A8@@IHAAHXZ@1@A"

using J = int ();
S<J> j;
// CHECK-DAG: @"?j@FTypeWithQuals@@3U?$S@$$A6AHXZ@1@A"

using K = int () &;
S<K> k;
// CHECK-DAG: @"?k@FTypeWithQuals@@3U?$S@$$A8@@GAAHXZ@1@A"

using L = int () &&;
S<L> l;
// CHECK-DAG: @"?l@FTypeWithQuals@@3U?$S@$$A8@@HAAHXZ@1@A"
}

// CHECK: "?DeducedType@@3HA"
auto DeducedType = 30;

// CHECK-DAG: @"?Char16Var@@3_SA"
char16_t Char16Var;

// CHECK-DAG: @"?Char32Var@@3_UA"
char32_t Char32Var;

// CHECK: "?LRef@@YAXAAH@Z"
void LRef(int& a) { }

// CHECK: "?RRef@@YAH$$QAH@Z"
int RRef(int&& a) { return a; }

// CHECK: "?Null@@YAX$$T@Z"
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
// CHECK-DAG: @"?Enum@EnumMangling@@3W4Enum01@1@A"
// CHECK-DAG: @"?BoolEnum@EnumMangling@@3W4Enum02@1@A
// CHECK-DAG: @"?CharEnum@EnumMangling@@3W4Enum03@1@A
// CHECK-DAG: @"?SCharEnum@EnumMangling@@3W4Enum04@1@A
// CHECK-DAG: @"?UCharEnum@EnumMangling@@3W4Enum05@1@A
// CHECK-DAG: @"?SShortEnum@EnumMangling@@3W4Enum06@1@A"
// CHECK-DAG: @"?UShortEnum@EnumMangling@@3W4Enum07@1@A"
// CHECK-DAG: @"?SIntEnum@EnumMangling@@3W4Enum08@1@A"
// CHECK-DAG: @"?UIntEnum@EnumMangling@@3W4Enum09@1@A"
// CHECK-DAG: @"?SLongEnum@EnumMangling@@3W4Enum10@1@A"
// CHECK-DAG: @"?ULongEnum@EnumMangling@@3W4Enum11@1@A"
// CHECK-DAG: @"?SLongLongEnum@EnumMangling@@3W4Enum12@1@A"
// CHECK-DAG: @"?ULongLongEnum@EnumMangling@@3W4Enum13@1@A"
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
// CHECK-DAG: @"?EnumClass@EnumMangling@@3W4EnumClass01@1@A"
// CHECK-DAG: @"?BoolEnumClass@EnumMangling@@3W4EnumClass02@1@A
// CHECK-DAG: @"?CharEnumClass@EnumMangling@@3W4EnumClass03@1@A
// CHECK-DAG: @"?SCharEnumClass@EnumMangling@@3W4EnumClass04@1@A
// CHECK-DAG: @"?UCharEnumClass@EnumMangling@@3W4EnumClass05@1@A
// CHECK-DAG: @"?SShortEnumClass@EnumMangling@@3W4EnumClass06@1@A"
// CHECK-DAG: @"?UShortEnumClass@EnumMangling@@3W4EnumClass07@1@A"
// CHECK-DAG: @"?SIntEnumClass@EnumMangling@@3W4EnumClass08@1@A"
// CHECK-DAG: @"?UIntEnumClass@EnumMangling@@3W4EnumClass09@1@A"
// CHECK-DAG: @"?SLongEnumClass@EnumMangling@@3W4EnumClass10@1@A"
// CHECK-DAG: @"?ULongEnumClass@EnumMangling@@3W4EnumClass11@1@A"
// CHECK-DAG: @"?SLongLongEnumClass@EnumMangling@@3W4EnumClass12@1@A"
// CHECK-DAG: @"?ULongLongEnumClass@EnumMangling@@3W4EnumClass13@1@A"
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
// CHECK-DAG: @"?fun@PR18022@@YA?AU<unnamed-type-a>@1@U21@0@Z"

void use_fun() { fun(a, a); }

}

inline int define_lambda() {
  static auto lambda = [] { static int local; ++local; return local; };
// First, we have the static local variable of type "<lambda_1>" inside of
// "define_lambda".
// CHECK-DAG: @"?lambda@?1??define_lambda@@YAHXZ@4V<lambda_1>@?0??1@YAHXZ@A"
// Next, we have the "operator()" for "<lambda_1>" which is inside of
// "define_lambda".
// CHECK-DAG: @"??R<lambda_1>@?0??define_lambda@@YAHXZ@QBE@XZ"
// Finally, we have the local which is inside of "<lambda_1>" which is inside of
// "define_lambda". Hooray.
// MSVC2013-DAG: @"?local@?2???R<lambda_1>@?0??define_lambda@@YAHXZ@QBE@XZ@4HA"
// MSVC2015-DAG: @"?local@?1???R<lambda_1>@?0??define_lambda@@YAHXZ@QBE@XZ@4HA"
  return lambda();
}

template <typename T>
void use_lambda_arg(T) {}

inline void call_with_lambda_arg1() {
  use_lambda_arg([]{});
  // CHECK-DAG: @"??$use_lambda_arg@V<lambda_1>@?0??call_with_lambda_arg1@@YAXXZ@@@YAXV<lambda_1>@?0??call_with_lambda_arg1@@YAXXZ@@Z"
}

inline void call_with_lambda_arg2() {
  use_lambda_arg([]{});
  // CHECK-DAG: @"??$use_lambda_arg@V<lambda_1>@?0??call_with_lambda_arg2@@YAXXZ@@@YAXV<lambda_1>@?0??call_with_lambda_arg2@@YAXXZ@@Z"
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
// CHECK-DAG: @"?foo@A@PR19361@@QIGAEXXZ"
void A::foo() __restrict && {}
// CHECK-DAG: @"?foo@A@PR19361@@QIHAEXXZ"
}

int operator"" _deg(long double) { return 0; }
// CHECK-DAG: @"??__K_deg@@YAHO@Z"

template <char...>
void templ_fun_with_pack() {}

template void templ_fun_with_pack<>();
// CHECK-DAG: @"??$templ_fun_with_pack@$S@@YAXXZ"

template <typename...>
void templ_fun_with_ty_pack() {}

template void templ_fun_with_ty_pack<>();
// MSVC2013-DAG: @"??$templ_fun_with_ty_pack@$$$V@@YAXXZ"
// MSVC2015-DAG: @"??$templ_fun_with_ty_pack@$$V@@YAXXZ"

template <template <class> class...>
void templ_fun_with_templ_templ_pack() {}

template void templ_fun_with_templ_templ_pack<>();
// MSVC2013-DAG: @"??$templ_fun_with_templ_templ_pack@$$$V@@YAXXZ"
// MSVC2015-DAG: @"??$templ_fun_with_templ_templ_pack@$$V@@YAXXZ"

namespace PR20047 {
template <typename T>
struct A {};

template <typename T>
using AliasA = A<T>;

template <template <typename> class>
void f() {}

template void f<AliasA>();
// CHECK-DAG: @"??$f@$$YAliasA@PR20047@@@PR20047@@YAXXZ"
}

namespace UnnamedType {
struct A {
  struct {} *TD;
};

void f(decltype(*A::TD)) {}
// CHECK-DAG: @"?f@UnnamedType@@YAXAAU<unnamed-type-TD>@A@1@@Z"

template <typename T>
struct B {
  enum {
  } *e;
};

void f(decltype(B<int>::e)) {}
// CHECK-DAG: @"?f@UnnamedType@@YAXPAW4<unnamed-type-e>@?$B@H@1@@Z
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
// CHECK-DAG: @"??$f@W4<unnamed-type-E>@?1??g@PR24651@@YAXXZ@@PR24651@@YAXW4<unnamed-type-E>@?1??g@0@YAXXZ@@Z"
// CHECK-DAG: @"??$f@W4<unnamed-type-E>@?2??g@PR24651@@YAXXZ@@PR24651@@YAXW4<unnamed-type-E>@?2??g@0@YAXXZ@@Z"
}

namespace PR18204 {
template <typename T>
int f(T *) { return 0; }
static union {
  int n = f(this);
};
// CHECK-DAG: @"??$f@T<unnamed-type-$S1>@PR18204@@@PR18204@@YAHPAT<unnamed-type-$S1>@0@@Z"
}

int PR26105() {
  auto add = [](int x) { return ([x](int y) { return x + y; }); };
  return add(3)(4);
}
// CHECK-DAG: @"??R<lambda_0>@?0??PR26105@@YAHXZ@QBE@H@Z"
// CHECK-DAG: @"??R<lambda_1>@?0???R<lambda_0>@?0??PR26105@@YAHXZ@QBE@H@Z@QBE@H@Z"

int __unaligned * unaligned_foo1() { return 0; }
int __unaligned * __unaligned * unaligned_foo2() { return 0; }
__unaligned int unaligned_foo3() { return 0; }
void unaligned_foo4(int __unaligned *p1) {}
void unaligned_foo5(int __unaligned * __restrict p1) {}
template <typename T> T unaligned_foo6(T t) { return t; }
void unaligned_foo7() { unaligned_foo6<int *>(0); unaligned_foo6<int __unaligned *>(0); }

// CHECK-DAG: @"?unaligned_foo1@@YAPFAHXZ"
// CHECK-DAG: @"?unaligned_foo2@@YAPFAPFAHXZ"
// CHECK-DAG: @"?unaligned_foo3@@YAHXZ"
// CHECK-DAG: @"?unaligned_foo4@@YAXPFAH@Z"
// CHECK-DAG: @"?unaligned_foo5@@YAXPIFAH@Z"
// CHECK-DAG: @"??$unaligned_foo6@PAH@@YAPAHPAH@Z"
// CHECK-DAG: @"??$unaligned_foo6@PFAH@@YAPFAHPFAH@Z"

// __unaligned qualifier for function types
struct unaligned_foo8_S {
    void unaligned_foo8() volatile __unaligned;
};
void unaligned_foo8_S::unaligned_foo8() volatile __unaligned {}

// CHECK-DAG: @"?unaligned_foo8@unaligned_foo8_S@@QFCEXXZ"

namespace PR31197 {
struct A {
  // CHECK-DAG: define linkonce_odr dso_local x86_thiscallcc noundef ptr @"??R<lambda_1>@x@A@PR31197@@QBE@XZ"(
  int *x = []() {
    static int white;
    // CHECK-DAG: @"?white@?1???R<lambda_1>@x@A@PR31197@@QBE@XZ@4HA"
    return &white;
  }();
  // CHECK-DAG: define linkonce_odr dso_local x86_thiscallcc noundef ptr @"??R<lambda_1>@y@A@PR31197@@QBE@XZ"(
  int *y = []() {
    static int black;
    // CHECK-DAG: @"?black@?1???R<lambda_1>@y@A@PR31197@@QBE@XZ@4HA"
    return &black;
  }();
  using FPtrTy = void(void);
  static void default_args(FPtrTy x = [] {}, FPtrTy y = [] {}, int z = [] { return 1; }() + [] { return 2; }()) {}
  // CHECK-DAG: @"??R<lambda_1_1>@?0??default_args@A@PR31197@@SAXP6AXXZ0H@Z@QBE@XZ"(
  // CHECK-DAG: @"??R<lambda_1_2>@?0??default_args@A@PR31197@@SAXP6AXXZ0H@Z@QBE@XZ"(
  // CHECK-DAG: @"??R<lambda_2_1>@?0??default_args@A@PR31197@@SAXP6AXXZ0H@Z@QBE@XZ"(
  // CHECK-DAG: @"??R<lambda_3_1>@?0??default_args@A@PR31197@@SAXP6AXXZ0H@Z@QBE@XZ"(
};
A a;

int call_it = (A::default_args(), 1);
}

enum { enumerator };
void f(decltype(enumerator)) {}
// CHECK-DAG: define internal void @"?f@@YAXW4<unnamed-enum-enumerator>@@@Z"(
void use_f() { f(enumerator); }

namespace pr37723 {
struct s { enum {}; enum {}; };
// DBG-DAG: DW_TAG_enumeration_type{{.*}}identifier: ".?AW4<unnamed-type-$S2>@s@pr37723@@"
// DBG-DAG: DW_TAG_enumeration_type{{.*}}identifier: ".?AW4<unnamed-type-$S3>@s@pr37723@@"
s x;
}
