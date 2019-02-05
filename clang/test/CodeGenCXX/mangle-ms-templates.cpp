// RUN: %clang_cc1 -std=c++11 -fms-compatibility-version=19 -emit-llvm %s -o - -fms-extensions -fdelayed-template-parsing -triple=i386-pc-win32 | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -fms-compatibility-version=19 -emit-llvm %s -o - -fms-extensions -fdelayed-template-parsing -triple=x86_64-pc-win32 | FileCheck -check-prefix X64 %s
// RUN: %clang_cc1 -std=c++17 -fms-compatibility-version=19 -emit-llvm %s -o - -fms-extensions -fdelayed-template-parsing -triple=i386-pc-win32 | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -fms-compatibility-version=19 -emit-llvm %s -o - -fms-extensions -fdelayed-template-parsing -triple=x86_64-pc-win32 | FileCheck -check-prefix X64 %s

template<typename T>
class Class {
 public:
  Class() {}
};

class Typename { };

template<typename T>
class Nested { };

template<bool flag>
class BoolTemplate {
 public:
  BoolTemplate() {}
};

template<int param>
class IntTemplate {
 public:
  IntTemplate() {}
};

template<unsigned param>
class UnsignedIntTemplate {
public:
  UnsignedIntTemplate() {}
};

template<long long param>
class LongLongTemplate {
 public:
  LongLongTemplate() {}
};

template<unsigned long long param>
class UnsignedLongLongTemplate {
 public:
  UnsignedLongLongTemplate() {}
};

template<>
class BoolTemplate<true> {
 public:
  BoolTemplate() {}
  template<class T> void Foo(T arg) {}
};

void template_mangling() {
  Class<Typename> c1;
// CHECK: call {{.*}} @"??0?$Class@VTypename@@@@QAE@XZ"
// X64: call {{.*}} @"??0?$Class@VTypename@@@@QEAA@XZ"

  Class<const Typename> c1_const;
// CHECK: call {{.*}} @"??0?$Class@$$CBVTypename@@@@QAE@XZ"
// X64: call {{.*}} @"??0?$Class@$$CBVTypename@@@@QEAA@XZ"
  Class<volatile Typename> c1_volatile;
// CHECK: call {{.*}} @"??0?$Class@$$CCVTypename@@@@QAE@XZ"
// X64: call {{.*}} @"??0?$Class@$$CCVTypename@@@@QEAA@XZ"
  Class<const volatile Typename> c1_cv;
// CHECK: call {{.*}} @"??0?$Class@$$CDVTypename@@@@QAE@XZ"
// X64: call {{.*}} @"??0?$Class@$$CDVTypename@@@@QEAA@XZ"

  Class<Nested<Typename> > c2;
// CHECK: call {{.*}} @"??0?$Class@V?$Nested@VTypename@@@@@@QAE@XZ"
// X64: call {{.*}} @"??0?$Class@V?$Nested@VTypename@@@@@@QEAA@XZ"

  Class<int * const> c_intpc;
// CHECK: call {{.*}} @"??0?$Class@QAH@@QAE@XZ"
// X64: call {{.*}} @"??0?$Class@QEAH@@QEAA@XZ"
  Class<int()> c_ft;
// CHECK: call {{.*}} @"??0?$Class@$$A6AHXZ@@QAE@XZ"
// X64: call {{.*}} @"??0?$Class@$$A6AHXZ@@QEAA@XZ"
  Class<int[]> c_inti;
// CHECK: call {{.*}} @"??0?$Class@$$BY0A@H@@QAE@XZ"
// X64: call {{.*}} @"??0?$Class@$$BY0A@H@@QEAA@XZ"
  Class<int[5]> c_int5;
// CHECK: call {{.*}} @"??0?$Class@$$BY04H@@QAE@XZ"
// X64: call {{.*}} @"??0?$Class@$$BY04H@@QEAA@XZ"
  Class<const int[5]> c_intc5;
// CHECK: call {{.*}} @"??0?$Class@$$BY04$$CBH@@QAE@XZ"
// X64: call {{.*}} @"??0?$Class@$$BY04$$CBH@@QEAA@XZ"
  Class<int * const[5]> c_intpc5;
// CHECK: call {{.*}} @"??0?$Class@$$BY04QAH@@QAE@XZ"
// X64: call {{.*}} @"??0?$Class@$$BY04QEAH@@QEAA@XZ"

  BoolTemplate<false> _false;
// CHECK: call {{.*}} @"??0?$BoolTemplate@$0A@@@QAE@XZ"
// X64: call {{.*}} @"??0?$BoolTemplate@$0A@@@QEAA@XZ"

  BoolTemplate<true> _true;
  // PR13158
  _true.Foo(1);
// CHECK: call {{.*}} @"??0?$BoolTemplate@$00@@QAE@XZ"
// X64: call {{.*}} @"??0?$BoolTemplate@$00@@QEAA@XZ"
// CHECK: call {{.*}} @"??$Foo@H@?$BoolTemplate@$00@@QAEXH@Z"
// X64: call {{.*}} @"??$Foo@H@?$BoolTemplate@$00@@QEAAXH@Z"

  IntTemplate<0> zero;
// CHECK: call {{.*}} @"??0?$IntTemplate@$0A@@@QAE@XZ"
// X64: call {{.*}} @"??0?$IntTemplate@$0A@@@QEAA@XZ"

  IntTemplate<5> five;
// CHECK: call {{.*}} @"??0?$IntTemplate@$04@@QAE@XZ"
// X64: call {{.*}} @"??0?$IntTemplate@$04@@QEAA@XZ"

  IntTemplate<11> eleven;
// CHECK: call {{.*}} @"??0?$IntTemplate@$0L@@@QAE@XZ"
// X64: call {{.*}} @"??0?$IntTemplate@$0L@@@QEAA@XZ"

  IntTemplate<256> _256;
// CHECK: call {{.*}} @"??0?$IntTemplate@$0BAA@@@QAE@XZ"
// X64: call {{.*}} @"??0?$IntTemplate@$0BAA@@@QEAA@XZ"

  IntTemplate<513> _513;
// CHECK: call {{.*}} @"??0?$IntTemplate@$0CAB@@@QAE@XZ"
// X64: call {{.*}} @"??0?$IntTemplate@$0CAB@@@QEAA@XZ"

  IntTemplate<1026> _1026;
// CHECK: call {{.*}} @"??0?$IntTemplate@$0EAC@@@QAE@XZ"
// X64: call {{.*}} @"??0?$IntTemplate@$0EAC@@@QEAA@XZ"

  IntTemplate<65535> ffff;
// CHECK: call {{.*}} @"??0?$IntTemplate@$0PPPP@@@QAE@XZ"
// X64: call {{.*}} @"??0?$IntTemplate@$0PPPP@@@QEAA@XZ"

  IntTemplate<-1>  neg_1;
// CHECK: call {{.*}} @"??0?$IntTemplate@$0?0@@QAE@XZ"
// X64: call {{.*}} @"??0?$IntTemplate@$0?0@@QEAA@XZ"
  IntTemplate<-9>  neg_9;
// CHECK: call {{.*}} @"??0?$IntTemplate@$0?8@@QAE@XZ"
// X64: call {{.*}} @"??0?$IntTemplate@$0?8@@QEAA@XZ"
  IntTemplate<-10> neg_10;
// CHECK: call {{.*}} @"??0?$IntTemplate@$0?9@@QAE@XZ"
// X64: call {{.*}} @"??0?$IntTemplate@$0?9@@QEAA@XZ"
  IntTemplate<-11> neg_11;
// CHECK: call {{.*}} @"??0?$IntTemplate@$0?L@@@QAE@XZ"
// X64: call {{.*}} @"??0?$IntTemplate@$0?L@@@QEAA@XZ"
  
  UnsignedIntTemplate<4294967295> ffffffff;
// CHECK: call {{.*}} @"??0?$UnsignedIntTemplate@$0PPPPPPPP@@@QAE@XZ"
// X64: call {{.*}} @"??0?$UnsignedIntTemplate@$0PPPPPPPP@@@QEAA@XZ"

  LongLongTemplate<-9223372036854775807LL-1LL> int64_min;
// CHECK: call {{.*}} @"??0?$LongLongTemplate@$0?IAAAAAAAAAAAAAAA@@@QAE@XZ"
// X64: call {{.*}} @"??0?$LongLongTemplate@$0?IAAAAAAAAAAAAAAA@@@QEAA@XZ"
  LongLongTemplate<9223372036854775807LL>      int64_max;
// CHECK: call {{.*}} @"??0?$LongLongTemplate@$0HPPPPPPPPPPPPPPP@@@QAE@XZ"
// X64: call {{.*}} @"??0?$LongLongTemplate@$0HPPPPPPPPPPPPPPP@@@QEAA@XZ"
  UnsignedLongLongTemplate<18446744073709551615ULL> uint64_max;
// CHECK: call {{.*}} @"??0?$UnsignedLongLongTemplate@$0?0@@QAE@XZ"
// X64: call {{.*}} @"??0?$UnsignedLongLongTemplate@$0?0@@QEAA@XZ"
  UnsignedLongLongTemplate<(unsigned long long)-1>  uint64_neg_1;
// CHECK: call {{.*}} @"??0?$UnsignedLongLongTemplate@$0?0@@QAE@XZ"
// X64: call {{.*}} @"??0?$UnsignedLongLongTemplate@$0?0@@QEAA@XZ"
}

namespace space {
  template<class T> const T& foo(const T& l) { return l; }
}
// CHECK: "??$foo@H@space@@YAABHABH@Z"
// X64: "??$foo@H@space@@YAAEBHAEBH@Z"

void use() {
  space::foo(42);
}

// PR13455
typedef void (*FunctionPointer)(void);

template <FunctionPointer function>
void FunctionPointerTemplate() {
  function();
}

void spam() {
  FunctionPointerTemplate<spam>();
// CHECK: "??$FunctionPointerTemplate@$1?spam@@YAXXZ@@YAXXZ"
// X64: "??$FunctionPointerTemplate@$1?spam@@YAXXZ@@YAXXZ"
}

// Unlike Itanium, there is no character code to indicate an argument pack.
// Tested with MSVC 2013, the first version which supports variadic templates.

template <typename ...Ts> void variadic_fn_template(const Ts &...args);
template <typename... Ts, typename... Us>
void multi_variadic_fn(Ts... ts, Us... us);
template <typename... Ts, typename C, typename... Us>
void multi_variadic_mixed(Ts... ts, C c, Us... us);
void variadic_fn_instantiate() {
  variadic_fn_template(0, 1, 3, 4);
  variadic_fn_template(0, 1, 'a', "b");

  // Directlly consecutive packs are separated by $$Z...
  multi_variadic_fn<int, int>(1, 2, 3, 4, 5);
  multi_variadic_fn<int, int, int>(1, 2, 3, 4, 5);

  // ...but not if another template parameter is between them.
  multi_variadic_mixed<int, int>(1, 2, 3);
  multi_variadic_mixed<int, int>(1, 2, 3, 4);
}
// CHECK: "??$variadic_fn_template@HHHH@@YAXABH000@Z"
// X64:   "??$variadic_fn_template@HHHH@@YAXAEBH000@Z"
// CHECK: "??$variadic_fn_template@HHD$$BY01D@@YAXABH0ABDAAY01$$CBD@Z"
// X64:   "??$variadic_fn_template@HHD$$BY01D@@YAXAEBH0AEBDAEAY01$$CBD@Z"
// CHECK: "??$multi_variadic_fn@HH$$ZHHH@@YAXHHHHH@Z"
// X64:   "??$multi_variadic_fn@HH$$ZHHH@@YAXHHHHH@Z"
// CHECK: "??$multi_variadic_fn@HHH$$ZHH@@YAXHHHHH@Z"
// X64:   "??$multi_variadic_fn@HHH$$ZHH@@YAXHHHHH@Z"
// CHECK: "??$multi_variadic_mixed@HHH$$V@@YAXHHH@Z"
// X64:   "??$multi_variadic_mixed@HHH$$V@@YAXHHH@Z"
// CHECK: "??$multi_variadic_mixed@HHHH@@YAXHHHH@Z"
// X64:   "??$multi_variadic_mixed@HHHH@@YAXHHHH@Z"

template <typename ...Ts>
struct VariadicClass {
  VariadicClass() { }
  int x;
};
void variadic_class_instantiate() {
  VariadicClass<int, char, bool> a;
  VariadicClass<bool, char, int> b;
}
// CHECK: call {{.*}} @"??0?$VariadicClass@HD_N@@QAE@XZ"
// CHECK: call {{.*}} @"??0?$VariadicClass@_NDH@@QAE@XZ"

template <typename T>
struct Second {};

template <typename T, template <class> class>
struct Type {};

template <template <class> class T>
struct Type2 {};

template <template <class> class T, bool B>
struct Thing;

template <template <class> class T>
struct Thing<T, false> { };

template <template <class> class T>
struct Thing<T, true> { };

void template_template_fun(Type<Thing<Second, true>, Second>) { }
// CHECK: "?template_template_fun@@YAXU?$Type@U?$Thing@USecond@@$00@@USecond@@@@@Z"

template <typename T>
void template_template_specialization();

template <>
void template_template_specialization<void (Type<Thing<Second, true>, Second>)>() {
}
// CHECK: "??$template_template_specialization@$$A6AXU?$Type@U?$Thing@USecond@@$00@@USecond@@@@@Z@@YAXXZ"

// PR16788
template <decltype(nullptr)> struct S1 {};
void f(S1<nullptr>) {}
// CHECK: "?f@@YAXU?$S1@$0A@@@@Z"

struct record {
  int first;
  int second;
};
template <const record &>
struct type1 {
};
extern const record inst;
void recref(type1<inst>) {}
// CHECK: "?recref@@YAXU?$type1@$E?inst@@3Urecord@@B@@@Z"

struct _GUID {};
struct __declspec(uuid("{12345678-1234-1234-1234-1234567890aB}")) uuid;

template <typename T, const _GUID *G = &__uuidof(T)>
struct UUIDType1 {};

template <typename T, const _GUID &G = __uuidof(T)>
struct UUIDType2 {};

void fun(UUIDType1<uuid> a) {}
// CHECK: "?fun@@YAXU?$UUIDType1@Uuuid@@$1?_GUID_12345678_1234_1234_1234_1234567890ab@@3U__s_GUID@@B@@@Z"
void fun(UUIDType2<uuid> b) {}
// CHECK: "?fun@@YAXU?$UUIDType2@Uuuid@@$E?_GUID_12345678_1234_1234_1234_1234567890ab@@3U__s_GUID@@B@@@Z"

template <typename T> struct TypeWithFriendDefinition {
  friend void FunctionDefinedWithInjectedName(TypeWithFriendDefinition<T>) {}
};
// CHECK: call {{.*}} @"?FunctionDefinedWithInjectedName@@YAXU?$TypeWithFriendDefinition@H@@@Z"
void CallFunctionDefinedWithInjectedName() {
  FunctionDefinedWithInjectedName(TypeWithFriendDefinition<int>());
}
// CHECK: @"?FunctionDefinedWithInjectedName@@YAXU?$TypeWithFriendDefinition@H@@@Z"

// We need to be able to feed GUIDs through a couple rounds of template
// substitution.
template <const _GUID *G>
struct UUIDType3 {
  void foo() {}
};
template <const _GUID *G>
struct UUIDType4 : UUIDType3<G> {
  void bar() { UUIDType4::foo(); }
};
template struct UUIDType4<&__uuidof(uuid)>;
// CHECK: "?bar@?$UUIDType4@$1?_GUID_12345678_1234_1234_1234_1234567890ab@@3U__s_GUID@@B@@QAEXXZ"
// CHECK: "?foo@?$UUIDType3@$1?_GUID_12345678_1234_1234_1234_1234567890ab@@3U__s_GUID@@B@@QAEXXZ"
