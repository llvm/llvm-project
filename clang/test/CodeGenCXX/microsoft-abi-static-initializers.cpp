// RUN: %clang_cc1 -no-opaque-pointers -fms-extensions -fno-threadsafe-statics -emit-llvm %s -o - -mconstructor-aliases -triple=i386-pc-win32 | FileCheck %s

// CHECK: @llvm.global_ctors = appending global [5 x { i32, void ()*, i8* }] [
// CHECK: { i32, void ()*, i8* } { i32 65535, void ()* @"??__Eselectany1@@YAXXZ", i8* getelementptr inbounds (%struct.S, %struct.S* @"?selectany1@@3US@@A", i32 0, i32 0) },
// CHECK: { i32, void ()*, i8* } { i32 65535, void ()* @"??__Eselectany2@@YAXXZ", i8* getelementptr inbounds (%struct.S, %struct.S* @"?selectany2@@3US@@A", i32 0, i32 0) },
// CHECK: { i32, void ()*, i8* } { i32 65535, void ()* @"??__E?s@?$ExportedTemplate@H@@2US@@A@@YAXXZ", i8* getelementptr inbounds (%struct.S, %struct.S* @"?s@?$ExportedTemplate@H@@2US@@A", i32 0, i32 0) },
// CHECK: { i32, void ()*, i8* } { i32 65535, void ()* @"??__E?foo@?$B@H@@2VA@@A@@YAXXZ", i8* bitcast (%class.A* @"?foo@?$B@H@@2VA@@A" to i8*) },
// CHECK: { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_microsoft_abi_static_initializers.cpp, i8* null }
// CHECK: ]

struct S {
  S();
  ~S();
};

S s;

// CHECK: define internal void @"??__Es@@YAXXZ"()
// CHECK: call x86_thiscallcc noundef %struct.S* @"??0S@@QAE@XZ"
// CHECK: call i32 @atexit(void ()* @"??__Fs@@YAXXZ")
// CHECK: ret void

// CHECK: define internal void @"??__Fs@@YAXXZ"()
// CHECK: call x86_thiscallcc void @"??1S@@QAE@XZ"
// CHECK: ret void

// These globals should have initializers comdat associative with the global.
// See @llvm.global_ctors above.
__declspec(selectany) S selectany1;
__declspec(selectany) S selectany2;
// CHECK: define linkonce_odr dso_local void @"??__Eselectany1@@YAXXZ"() {{.*}} comdat
// CHECK-NOT: @"??_Bselectany1
// CHECK: call x86_thiscallcc noundef %struct.S* @"??0S@@QAE@XZ"
// CHECK: ret void
// CHECK: define linkonce_odr dso_local void @"??__Eselectany2@@YAXXZ"() {{.*}} comdat
// CHECK-NOT: @"??_Bselectany2
// CHECK: call x86_thiscallcc noundef %struct.S* @"??0S@@QAE@XZ"
// CHECK: ret void

// The implicitly instantiated static data member should have initializer
// comdat associative with the global.
template <typename T> struct __declspec(dllexport) ExportedTemplate {
  static S s;
};
template <typename T> S ExportedTemplate<T>::s;
void useExportedTemplate(ExportedTemplate<int> x) {
  (void)x.s;
}

void StaticLocal() {
  static S TheS;
}

// CHECK-LABEL: define dso_local void @"?StaticLocal@@YAXXZ"()
// CHECK: load i32, i32* @"?$S1@?1??StaticLocal@@YAXXZ@4IA"
// CHECK: store i32 {{.*}}, i32* @"?$S1@?1??StaticLocal@@YAXXZ@4IA"
// CHECK: ret

void MultipleStatics() {
  static S S1;
  static S S2;
  static S S3;
  static S S4;
  static S S5;
  static S S6;
  static S S7;
  static S S8;
  static S S9;
  static S S10;
  static S S11;
  static S S12;
  static S S13;
  static S S14;
  static S S15;
  static S S16;
  static S S17;
  static S S18;
  static S S19;
  static S S20;
  static S S21;
  static S S22;
  static S S23;
  static S S24;
  static S S25;
  static S S26;
  static S S27;
  static S S28;
  static S S29;
  static S S30;
  static S S31;
  static S S32;
  static S S33;
  static S S34;
  static S S35;
}
// CHECK-LABEL: define dso_local void @"?MultipleStatics@@YAXXZ"()
// CHECK: load i32, i32* @"?$S1@?1??MultipleStatics@@YAXXZ@4IA"
// CHECK: and i32 {{.*}}, 1
// CHECK: and i32 {{.*}}, 2
// CHECK: and i32 {{.*}}, 4
// CHECK: and i32 {{.*}}, 8
// CHECK: and i32 {{.*}}, 16
//   ...
// CHECK: and i32 {{.*}}, -2147483648
// CHECK: load i32, i32* @"?$S1@?1??MultipleStatics@@YAXXZ@4IA.1"
// CHECK: and i32 {{.*}}, 1
// CHECK: and i32 {{.*}}, 2
// CHECK: and i32 {{.*}}, 4
// CHECK: ret

// Force WeakODRLinkage by using templates
class A {
 public:
  A() {}
  ~A() {}
  int a;
};

template<typename T>
class B {
 public:
  static A foo;
};

template<typename T> A B<T>::foo;

inline S &UnreachableStatic() {
  if (0) {
    static S s; // bit 1
    return s;
  }
  static S s; // bit 2
  return s;
}

// CHECK-LABEL: define linkonce_odr dso_local noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %struct.S* @"?UnreachableStatic@@YAAAUS@@XZ"() {{.*}} comdat
// CHECK: and i32 {{.*}}, 2
// CHECK: or i32 {{.*}}, 2
// CHECK: ret

inline S &getS() {
  static S TheS;
  return TheS;
}

// CHECK-LABEL: define linkonce_odr dso_local noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %struct.S* @"?getS@@YAAAUS@@XZ"() {{.*}} comdat
// CHECK: load i32, i32* @"??_B?1??getS@@YAAAUS@@XZ@51"
// CHECK: and i32 {{.*}}, 1
// CHECK: icmp eq i32 {{.*}}, 0
// CHECK: br i1
//   init:
// CHECK: or i32 {{.*}}, 1
// CHECK: store i32 {{.*}}, i32* @"??_B?1??getS@@YAAAUS@@XZ@51"
// CHECK: call x86_thiscallcc noundef %struct.S* @"??0S@@QAE@XZ"(%struct.S* {{[^,]*}} @"?TheS@?1??getS@@YAAAUS@@XZ@4U2@A")
// CHECK: call i32 @atexit(void ()* @"??__FTheS@?1??getS@@YAAAUS@@XZ@YAXXZ")
// CHECK: br label
//   init.end:
// CHECK: ret %struct.S* @"?TheS@?1??getS@@YAAAUS@@XZ@4U2@A"

inline int enum_in_function() {
  // CHECK-LABEL: define linkonce_odr dso_local noundef i32 @"?enum_in_function@@YAHXZ"() {{.*}} comdat
  static enum e { foo, bar, baz } x;
  // CHECK: @"?x@?1??enum_in_function@@YAHXZ@4W4e@?1??1@YAHXZ@A"
  static int y;
  // CHECK: @"?y@?1??enum_in_function@@YAHXZ@4HA"
  return x + y;
};

struct T {
  enum e { foo, bar, baz };
  int enum_in_struct() {
    // CHECK-LABEL: define linkonce_odr dso_local x86_thiscallcc noundef i32 @"?enum_in_struct@T@@QAEHXZ"({{.*}}) {{.*}} comdat
    static int x;
    // CHECK: @"?x@?1??enum_in_struct@T@@QAEHXZ@4HA"
    return x++;
  }
};

inline int switch_test(int x) {
  // CHECK-LABEL: define linkonce_odr dso_local noundef i32 @"?switch_test@@YAHH@Z"(i32 noundef %x) {{.*}} comdat
  switch (x) {
    static int a;
    // CHECK: @"?a@?3??switch_test@@YAHH@Z@4HA"
    case 0:
      a++;
      return 1;
    case 1:
      static int b;
      // CHECK: @"?b@?3??switch_test@@YAHH@Z@4HA"
      return b++;
    case 2: {
      static int c;
      // CHECK: @"?c@?4??switch_test@@YAHH@Z@4HA"
      return b + c++;
    }
  };
}

int f();
inline void switch_test2() {
  // CHECK-LABEL: define linkonce_odr dso_local void @"?switch_test2@@YAXXZ"() {{.*}} comdat
  // CHECK: @"?x@?2??switch_test2@@YAXXZ@4HA"
  switch (1) default: static int x = f();
}

namespace DynamicDLLImportInitVSMangling {
  // Failing to pop the ExprEvalContexts when instantiating a dllimport var with
  // dynamic initializer would cause subsequent static local numberings to be
  // incorrect.
  struct NonPOD { NonPOD(); };
  template <typename T> struct A { static NonPOD x; };
  template <typename T> NonPOD A<T>::x;
  template struct __declspec(dllimport) A<int>;

  inline int switch_test3() {
    // CHECK-LABEL: define linkonce_odr dso_local noundef i32 @"?switch_test3@DynamicDLLImportInitVSMangling@@YAHXZ"() {{.*}} comdat
    static int local;
    // CHECK: @"?local@?1??switch_test3@DynamicDLLImportInitVSMangling@@YAHXZ@4HA"
    return local++;
  }
}

void force_usage() {
  UnreachableStatic();
  getS();
  (void)B<int>::foo;  // (void) - force usage
  enum_in_function();
  (void)&T::enum_in_struct;
  switch_test(1);
  switch_test2();
  DynamicDLLImportInitVSMangling::switch_test3();
}

// CHECK: define linkonce_odr dso_local void @"??__E?foo@?$B@H@@2VA@@A@@YAXXZ"() {{.*}} comdat
// CHECK-NOT: and
// CHECK-NOT: ?_Bfoo@
// CHECK: call x86_thiscallcc noundef %class.A* @"??0A@@QAE@XZ"
// CHECK: call i32 @atexit(void ()* @"??__F?foo@?$B@H@@2VA@@A@@YAXXZ")
// CHECK: ret void

// CHECK: define linkonce_odr dso_local x86_thiscallcc noundef %class.A* @"??0A@@QAE@XZ"({{.*}}) {{.*}} comdat

// CHECK: define linkonce_odr dso_local x86_thiscallcc void @"??1A@@QAE@XZ"({{.*}}) {{.*}} comdat

// CHECK: define internal void @"??__F?foo@?$B@H@@2VA@@A@@YAXXZ"
// CHECK: call x86_thiscallcc void @"??1A@@QAE@XZ"{{.*}}foo
// CHECK: ret void

// CHECK: define internal void @_GLOBAL__sub_I_microsoft_abi_static_initializers.cpp()
// CHECK: call void @"??__Es@@YAXXZ"()
// CHECK: ret void
