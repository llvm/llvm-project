// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=i386-pc-linux | FileCheck -check-prefix LINUX %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=i386-pc-win32 -mconstructor-aliases -fno-rtti | FileCheck -check-prefix WIN32 %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=thumb-pc-win32 -mconstructor-aliases -fno-rtti | FileCheck -check-prefix WOA %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=x86_64-pc-win32 -mconstructor-aliases -fno-rtti | FileCheck -check-prefix WIN64 %s

struct Empty {};

struct EmptyWithCtor {
  EmptyWithCtor() {}
};

struct Small {
  int x;
};

// This is a C++11 trivial and standard-layout struct but not a C++03 POD.
struct SmallCpp11NotCpp03Pod : Empty {
  int x;
};

struct SmallWithCtor {
  SmallWithCtor() {}
  int x;
};

struct Multibyte {
  char a, b, c, d;
};

struct Packed {
  short a;
  int b;
  short c;
};

struct SmallWithDtor {
  SmallWithDtor();
  ~SmallWithDtor();
  int x;
};

struct SmallWithVftable {
  int x;
  virtual void foo();
};

struct Medium {
  int x, y;
};

struct MediumWithCopyCtor {
  MediumWithCopyCtor();
  MediumWithCopyCtor(const struct MediumWithCopyCtor &);
  int x, y;
};

struct Big {
  int a, b, c, d, e, f;
};

struct BigWithDtor {
  BigWithDtor();
  ~BigWithDtor();
  int a, b, c, d, e, f;
};

struct BaseNoByval : Small {
  int bb;
};

// WIN32: declare dso_local void @"{{.*take_bools_and_chars.*}}"
// WIN32:       (<{ i8, [3 x i8], i8, [3 x i8], %struct.SmallWithDtor,
// WIN32:           i8, [3 x i8], i8, [3 x i8], i32, i8, [3 x i8] }>* inalloca)
void take_bools_and_chars(char a, char b, SmallWithDtor c, char d, bool e, int f, bool g);
void call_bools_and_chars() {
  take_bools_and_chars('A', 'B', SmallWithDtor(), 'D', true, 13, false);
}

// Returning structs that fit into a register.
Small small_return() { return Small(); }
// LINUX-LABEL: define void @_Z12small_returnv(%struct.Small* noalias sret %agg.result)
// WIN32: define dso_local i32 @"\01?small_return@@YA?AUSmall@@XZ"()
// WIN64: define dso_local i32 @"\01?small_return@@YA?AUSmall@@XZ"()

Medium medium_return() { return Medium(); }
// LINUX-LABEL: define void @_Z13medium_returnv(%struct.Medium* noalias sret %agg.result)
// WIN32: define dso_local i64 @"\01?medium_return@@YA?AUMedium@@XZ"()
// WIN64: define dso_local i64 @"\01?medium_return@@YA?AUMedium@@XZ"()

// Returning structs that fit into a register but are not POD.
SmallCpp11NotCpp03Pod small_non_pod_return() { return SmallCpp11NotCpp03Pod(); }
// LINUX-LABEL: define void @_Z20small_non_pod_returnv(%struct.SmallCpp11NotCpp03Pod* noalias sret %agg.result)
// WIN32: define dso_local void @"\01?small_non_pod_return@@YA?AUSmallCpp11NotCpp03Pod@@XZ"(%struct.SmallCpp11NotCpp03Pod* noalias sret %agg.result)
// WIN64: define dso_local void @"\01?small_non_pod_return@@YA?AUSmallCpp11NotCpp03Pod@@XZ"(%struct.SmallCpp11NotCpp03Pod* noalias sret %agg.result)

SmallWithCtor small_with_ctor_return() { return SmallWithCtor(); }
// LINUX-LABEL: define void @_Z22small_with_ctor_returnv(%struct.SmallWithCtor* noalias sret %agg.result)
// WIN32: define dso_local void @"\01?small_with_ctor_return@@YA?AUSmallWithCtor@@XZ"(%struct.SmallWithCtor* noalias sret %agg.result)
// WIN64: define dso_local void @"\01?small_with_ctor_return@@YA?AUSmallWithCtor@@XZ"(%struct.SmallWithCtor* noalias sret %agg.result)
// FIXME: The 'sret' mark here doesn't seem to be enough to convince LLVM to
// preserve the hidden sret pointer in R0 across the function.
// WOA: define dso_local arm_aapcs_vfpcc void @"\01?small_with_ctor_return@@YA?AUSmallWithCtor@@XZ"(%struct.SmallWithCtor* noalias sret %agg.result)

SmallWithVftable small_with_vftable_return() { return SmallWithVftable(); }
// LINUX-LABEL: define void @_Z25small_with_vftable_returnv(%struct.SmallWithVftable* noalias sret %agg.result)
// WIN32: define dso_local void @"\01?small_with_vftable_return@@YA?AUSmallWithVftable@@XZ"(%struct.SmallWithVftable* noalias sret %agg.result)
// WIN64: define dso_local void @"\01?small_with_vftable_return@@YA?AUSmallWithVftable@@XZ"(%struct.SmallWithVftable* noalias sret %agg.result)

MediumWithCopyCtor medium_with_copy_ctor_return() { return MediumWithCopyCtor(); }
// LINUX-LABEL: define void @_Z28medium_with_copy_ctor_returnv(%struct.MediumWithCopyCtor* noalias sret %agg.result)
// WIN32: define dso_local void @"\01?medium_with_copy_ctor_return@@YA?AUMediumWithCopyCtor@@XZ"(%struct.MediumWithCopyCtor* noalias sret %agg.result)
// WIN64: define dso_local void @"\01?medium_with_copy_ctor_return@@YA?AUMediumWithCopyCtor@@XZ"(%struct.MediumWithCopyCtor* noalias sret %agg.result)
// WOA: define dso_local arm_aapcs_vfpcc void @"\01?medium_with_copy_ctor_return@@YA?AUMediumWithCopyCtor@@XZ"(%struct.MediumWithCopyCtor* noalias sret %agg.result)

// Returning a large struct that doesn't fit into a register.
Big big_return() { return Big(); }
// LINUX-LABEL: define void @_Z10big_returnv(%struct.Big* noalias sret %agg.result)
// WIN32: define dso_local void @"\01?big_return@@YA?AUBig@@XZ"(%struct.Big* noalias sret %agg.result)
// WIN64: define dso_local void @"\01?big_return@@YA?AUBig@@XZ"(%struct.Big* noalias sret %agg.result)


void small_arg(Small s) {}
// LINUX-LABEL: define void @_Z9small_arg5Small(i32 %s.0)
// WIN32: define dso_local void @"\01?small_arg@@YAXUSmall@@@Z"(i32 %s.0)
// WIN64: define dso_local void @"\01?small_arg@@YAXUSmall@@@Z"(i32 %s.coerce)
// WOA: define dso_local arm_aapcs_vfpcc void @"\01?small_arg@@YAXUSmall@@@Z"([1 x i32] %s.coerce)

void medium_arg(Medium s) {}
// LINUX-LABEL: define void @_Z10medium_arg6Medium(i32 %s.0, i32 %s.1)
// WIN32: define dso_local void @"\01?medium_arg@@YAXUMedium@@@Z"(i32 %s.0, i32 %s.1)
// WIN64: define dso_local void @"\01?medium_arg@@YAXUMedium@@@Z"(i64 %s.coerce)
// WOA: define dso_local arm_aapcs_vfpcc void @"\01?medium_arg@@YAXUMedium@@@Z"([2 x i32] %s.coerce)

void base_no_byval_arg(BaseNoByval s) {}
// LINUX-LABEL: define void @_Z17base_no_byval_arg11BaseNoByval(%struct.BaseNoByval* byval align 4 %s)
// WIN32: define dso_local void @"\01?base_no_byval_arg@@YAXUBaseNoByval@@@Z"(i32 %s.0, i32 %s.1)
// WIN64: define dso_local void @"\01?base_no_byval_arg@@YAXUBaseNoByval@@@Z"(i64 %s.coerce)
// WOA: define dso_local arm_aapcs_vfpcc void @"\01?base_no_byval_arg@@YAXUBaseNoByval@@@Z"([2 x i32] %s.coerce)

void small_arg_with_ctor(SmallWithCtor s) {}
// LINUX-LABEL: define void @_Z19small_arg_with_ctor13SmallWithCtor(%struct.SmallWithCtor* byval align 4 %s)
// WIN32: define dso_local void @"\01?small_arg_with_ctor@@YAXUSmallWithCtor@@@Z"(i32 %s.0)
// WIN64: define dso_local void @"\01?small_arg_with_ctor@@YAXUSmallWithCtor@@@Z"(i32 %s.coerce)
// WOA: define dso_local arm_aapcs_vfpcc void @"\01?small_arg_with_ctor@@YAXUSmallWithCtor@@@Z"([1 x i32] %s.coerce)

// FIXME: We could coerce to a series of i32s here if we wanted to.
void multibyte_arg(Multibyte s) {}
// LINUX-LABEL: define void @_Z13multibyte_arg9Multibyte(%struct.Multibyte* byval align 4 %s)
// WIN32: define dso_local void @"\01?multibyte_arg@@YAXUMultibyte@@@Z"(%struct.Multibyte* byval align 4 %s)
// WIN64: define dso_local void @"\01?multibyte_arg@@YAXUMultibyte@@@Z"(i32 %s.coerce)
// WOA: define dso_local arm_aapcs_vfpcc void @"\01?multibyte_arg@@YAXUMultibyte@@@Z"([1 x i32] %s.coerce)

void packed_arg(Packed s) {}
// LINUX-LABEL: define void @_Z10packed_arg6Packed(%struct.Packed* byval align 4 %s)
// WIN32: define dso_local void @"\01?packed_arg@@YAXUPacked@@@Z"(%struct.Packed* byval align 4 %s)
// WIN64: define dso_local void @"\01?packed_arg@@YAXUPacked@@@Z"(%struct.Packed* %s)

// Test that dtors are invoked in the callee.
void small_arg_with_dtor(SmallWithDtor s) {}
// WIN32: define dso_local void @"\01?small_arg_with_dtor@@YAXUSmallWithDtor@@@Z"(<{ %struct.SmallWithDtor }>* inalloca) {{.*}} {
// WIN32:   call x86_thiscallcc void @"\01??1SmallWithDtor@@QAE@XZ"
// WIN32: }
// WIN64: define dso_local void @"\01?small_arg_with_dtor@@YAXUSmallWithDtor@@@Z"(i32 %s.coerce) {{.*}} {
// WIN64:   call void @"\01??1SmallWithDtor@@QEAA@XZ"
// WIN64: }

// FIXME: MSVC incompatible!
// WOA: define dso_local arm_aapcs_vfpcc void @"\01?small_arg_with_dtor@@YAXUSmallWithDtor@@@Z"(%struct.SmallWithDtor* %s) {{.*}} {
// WOA:   call arm_aapcs_vfpcc void @"\01??1SmallWithDtor@@QAA@XZ"(%struct.SmallWithDtor* %s)
// WOA: }

void call_small_arg_with_dtor() {
  small_arg_with_dtor(SmallWithDtor());
}
// The temporary is copied, so it's destroyed in the caller as well as the
// callee.
// WIN64-LABEL: define dso_local void @"\01?call_small_arg_with_dtor@@YAXXZ"()
// WIN64:   call %struct.SmallWithDtor* @"\01??0SmallWithDtor@@QEAA@XZ"
// WIN64:   call void @"\01?small_arg_with_dtor@@YAXUSmallWithDtor@@@Z"(i32 %{{.*}})
// WIN64:   call void @"\01??1SmallWithDtor@@QEAA@XZ"
// WIN64:   ret void

// Test that references aren't destroyed in the callee.
void ref_small_arg_with_dtor(const SmallWithDtor &s) { }
// WIN32: define dso_local void @"\01?ref_small_arg_with_dtor@@YAXABUSmallWithDtor@@@Z"(%struct.SmallWithDtor* dereferenceable({{[0-9]+}}) %s) {{.*}} {
// WIN32-NOT:   call x86_thiscallcc void @"\01??1SmallWithDtor@@QAE@XZ"
// WIN32: }
// WIN64-LABEL: define dso_local void @"\01?ref_small_arg_with_dtor@@YAXAEBUSmallWithDtor@@@Z"(%struct.SmallWithDtor* dereferenceable({{[0-9]+}}) %s)

void big_arg_with_dtor(BigWithDtor s) {}
// WIN64-LABEL: define dso_local void @"\01?big_arg_with_dtor@@YAXUBigWithDtor@@@Z"(%struct.BigWithDtor* %s)
// WIN64:   call void @"\01??1BigWithDtor@@QEAA@XZ"
// WIN64: }

void call_big_arg_with_dtor() {
  big_arg_with_dtor(BigWithDtor());
}
// We can elide the copy of the temporary in the caller, because this object is
// larger than 8 bytes and is passed indirectly.
// WIN64-LABEL: define dso_local void @"\01?call_big_arg_with_dtor@@YAXXZ"()
// WIN64:   call %struct.BigWithDtor* @"\01??0BigWithDtor@@QEAA@XZ"
// WIN64:   call void @"\01?big_arg_with_dtor@@YAXUBigWithDtor@@@Z"(%struct.BigWithDtor* %{{.*}})
// WIN64-NOT: call void @"\01??1BigWithDtor@@QEAA@XZ"
// WIN64:   ret void

// Test that temporaries passed by reference are destroyed in the caller.
void temporary_ref_with_dtor() {
  ref_small_arg_with_dtor(SmallWithDtor());
}
// WIN32: define dso_local void @"\01?temporary_ref_with_dtor@@YAXXZ"() {{.*}} {
// WIN32:   call x86_thiscallcc %struct.SmallWithDtor* @"\01??0SmallWithDtor@@QAE@XZ"
// WIN32:   call void @"\01?ref_small_arg_with_dtor@@YAXABUSmallWithDtor@@@Z"
// WIN32:   call x86_thiscallcc void @"\01??1SmallWithDtor@@QAE@XZ"
// WIN32: }

void takes_two_by_val_with_dtor(SmallWithDtor a, SmallWithDtor b);
void eh_cleanup_arg_with_dtor() {
  takes_two_by_val_with_dtor(SmallWithDtor(), SmallWithDtor());
}
//   When exceptions are off, we don't have any cleanups.  See
//   microsoft-abi-exceptions.cpp for these cleanups.
// WIN32: define dso_local void @"\01?eh_cleanup_arg_with_dtor@@YAXXZ"() {{.*}} {
// WIN32:   call x86_thiscallcc %struct.SmallWithDtor* @"\01??0SmallWithDtor@@QAE@XZ"
// WIN32:   call x86_thiscallcc %struct.SmallWithDtor* @"\01??0SmallWithDtor@@QAE@XZ"
// WIN32:   call void @"\01?takes_two_by_val_with_dtor@@YAXUSmallWithDtor@@0@Z"
// WIN32-NOT: call x86_thiscallcc void @"\01??1SmallWithDtor@@QAE@XZ"
// WIN32: }

void small_arg_with_vftable(SmallWithVftable s) {}
// LINUX-LABEL: define void @_Z22small_arg_with_vftable16SmallWithVftable(%struct.SmallWithVftable* %s)
// WIN32: define dso_local void @"\01?small_arg_with_vftable@@YAXUSmallWithVftable@@@Z"(<{ %struct.SmallWithVftable }>* inalloca)
// WIN64: define dso_local void @"\01?small_arg_with_vftable@@YAXUSmallWithVftable@@@Z"(%struct.SmallWithVftable* %s)

void medium_arg_with_copy_ctor(MediumWithCopyCtor s) {}
// LINUX-LABEL: define void @_Z25medium_arg_with_copy_ctor18MediumWithCopyCtor(%struct.MediumWithCopyCtor* %s)
// WIN32: define dso_local void @"\01?medium_arg_with_copy_ctor@@YAXUMediumWithCopyCtor@@@Z"(<{ %struct.MediumWithCopyCtor }>* inalloca)
// WIN64: define dso_local void @"\01?medium_arg_with_copy_ctor@@YAXUMediumWithCopyCtor@@@Z"(%struct.MediumWithCopyCtor* %s)
// WOA: define dso_local arm_aapcs_vfpcc void @"\01?medium_arg_with_copy_ctor@@YAXUMediumWithCopyCtor@@@Z"(%struct.MediumWithCopyCtor* %s)

void big_arg(Big s) {}
// LINUX-LABEL: define void @_Z7big_arg3Big(%struct.Big* byval align 4 %s)
// WIN32: define dso_local void @"\01?big_arg@@YAXUBig@@@Z"(%struct.Big* byval align 4 %s)
// WIN64: define dso_local void @"\01?big_arg@@YAXUBig@@@Z"(%struct.Big* %s)

// PR27607: We would attempt to load i32 value out of the reference instead of
// just loading the pointer from the struct during argument expansion.
struct RefField {
  RefField(int &x);
  int &x;
};
void takes_ref_field(RefField s) {}
// LINUX-LABEL: define void @_Z15takes_ref_field8RefField(%struct.RefField* byval align 4 %s)
// WIN32: define dso_local void @"\01?takes_ref_field@@YAXURefField@@@Z"(i32* %s.0)
// WIN64: define dso_local void @"\01?takes_ref_field@@YAXURefField@@@Z"(i64 %s.coerce)

void pass_ref_field() {
  int x;
  takes_ref_field(RefField(x));
}
// LINUX-LABEL: define void @_Z14pass_ref_fieldv()
// LINUX: call void @_Z15takes_ref_field8RefField(%struct.RefField* byval align 4 %{{.*}})
// WIN32-LABEL: define dso_local void @"\01?pass_ref_field@@YAXXZ"()
// WIN32: call void @"\01?takes_ref_field@@YAXURefField@@@Z"(i32* %{{.*}})
// WIN64-LABEL: define dso_local void @"\01?pass_ref_field@@YAXXZ"()
// WIN64: call void @"\01?takes_ref_field@@YAXURefField@@@Z"(i64 %{{.*}})

class Class {
 public:
  Small thiscall_method_small() { return Small(); }
  // LINUX: define {{.*}} void @_ZN5Class21thiscall_method_smallEv(%struct.Small* noalias sret %agg.result, %class.Class* %this)
  // WIN32: define {{.*}} x86_thiscallcc void @"\01?thiscall_method_small@Class@@QAE?AUSmall@@XZ"(%class.Class* %this, %struct.Small* noalias sret %agg.result)
  // WIN64: define linkonce_odr dso_local void @"\01?thiscall_method_small@Class@@QEAA?AUSmall@@XZ"(%class.Class* %this, %struct.Small* noalias sret %agg.result)

  SmallWithCtor thiscall_method_small_with_ctor() { return SmallWithCtor(); }
  // LINUX: define {{.*}} void @_ZN5Class31thiscall_method_small_with_ctorEv(%struct.SmallWithCtor* noalias sret %agg.result, %class.Class* %this)
  // WIN32: define {{.*}} x86_thiscallcc void @"\01?thiscall_method_small_with_ctor@Class@@QAE?AUSmallWithCtor@@XZ"(%class.Class* %this, %struct.SmallWithCtor* noalias sret %agg.result)
  // WIN64: define linkonce_odr dso_local void @"\01?thiscall_method_small_with_ctor@Class@@QEAA?AUSmallWithCtor@@XZ"(%class.Class* %this, %struct.SmallWithCtor* noalias sret %agg.result)

  Small __cdecl cdecl_method_small() { return Small(); }
  // LINUX: define {{.*}} void @_ZN5Class18cdecl_method_smallEv(%struct.Small* noalias sret %agg.result, %class.Class* %this)
  // WIN32: define {{.*}} void @"\01?cdecl_method_small@Class@@QAA?AUSmall@@XZ"(%class.Class* %this, %struct.Small* noalias sret %agg.result)
  // WIN64: define linkonce_odr dso_local void @"\01?cdecl_method_small@Class@@QEAA?AUSmall@@XZ"(%class.Class* %this, %struct.Small* noalias sret %agg.result)

  Big __cdecl cdecl_method_big() { return Big(); }
  // LINUX: define {{.*}} void @_ZN5Class16cdecl_method_bigEv(%struct.Big* noalias sret %agg.result, %class.Class* %this)
  // WIN32: define {{.*}} void @"\01?cdecl_method_big@Class@@QAA?AUBig@@XZ"(%class.Class* %this, %struct.Big* noalias sret %agg.result)
  // WIN64: define linkonce_odr dso_local void @"\01?cdecl_method_big@Class@@QEAA?AUBig@@XZ"(%class.Class* %this, %struct.Big* noalias sret %agg.result)

  void thiscall_method_arg(Empty s) {}
  // LINUX: define {{.*}} void @_ZN5Class19thiscall_method_argE5Empty(%class.Class* %this)
  // WIN32: define {{.*}} void @"\01?thiscall_method_arg@Class@@QAEXUEmpty@@@Z"(%class.Class* %this, %struct.Empty* byval align 4 %s)
  // WIN64: define linkonce_odr dso_local void @"\01?thiscall_method_arg@Class@@QEAAXUEmpty@@@Z"(%class.Class* %this, i8 %s.coerce)

  void thiscall_method_arg(EmptyWithCtor s) {}
  // LINUX: define {{.*}} void @_ZN5Class19thiscall_method_argE13EmptyWithCtor(%class.Class* %this)
  // WIN32: define {{.*}} void @"\01?thiscall_method_arg@Class@@QAEXUEmptyWithCtor@@@Z"(%class.Class* %this, %struct.EmptyWithCtor* byval align 4 %s)
  // WIN64: define linkonce_odr dso_local void @"\01?thiscall_method_arg@Class@@QEAAXUEmptyWithCtor@@@Z"(%class.Class* %this, i8 %s.coerce)

  void thiscall_method_arg(Small s) {}
  // LINUX: define {{.*}} void @_ZN5Class19thiscall_method_argE5Small(%class.Class* %this, i32 %s.0)
  // WIN32: define {{.*}} void @"\01?thiscall_method_arg@Class@@QAEXUSmall@@@Z"(%class.Class* %this, i32 %s.0)
  // WIN64: define linkonce_odr dso_local void @"\01?thiscall_method_arg@Class@@QEAAXUSmall@@@Z"(%class.Class* %this, i32 %s.coerce)

  void thiscall_method_arg(SmallWithCtor s) {}
  // LINUX: define {{.*}} void @_ZN5Class19thiscall_method_argE13SmallWithCtor(%class.Class* %this, %struct.SmallWithCtor* byval align 4 %s)
  // WIN32: define {{.*}} void @"\01?thiscall_method_arg@Class@@QAEXUSmallWithCtor@@@Z"(%class.Class* %this, i32 %s.0)
  // WIN64: define linkonce_odr dso_local void @"\01?thiscall_method_arg@Class@@QEAAXUSmallWithCtor@@@Z"(%class.Class* %this, i32 %s.coerce)

  void thiscall_method_arg(Big s) {}
  // LINUX: define {{.*}} void @_ZN5Class19thiscall_method_argE3Big(%class.Class* %this, %struct.Big* byval align 4 %s)
  // WIN32: define {{.*}} void @"\01?thiscall_method_arg@Class@@QAEXUBig@@@Z"(%class.Class* %this, %struct.Big* byval align 4 %s)
  // WIN64: define linkonce_odr dso_local void @"\01?thiscall_method_arg@Class@@QEAAXUBig@@@Z"(%class.Class* %this, %struct.Big* %s)
};

void use_class() {
  Class c;
  c.thiscall_method_small();
  c.thiscall_method_small_with_ctor();

  c.cdecl_method_small();
  c.cdecl_method_big();

  c.thiscall_method_arg(Empty());
  c.thiscall_method_arg(EmptyWithCtor());
  c.thiscall_method_arg(Small());
  c.thiscall_method_arg(SmallWithCtor());
  c.thiscall_method_arg(Big());
}

struct X {
  X();
  ~X();
};
void g(X) {
}
// WIN32: define dso_local void @"\01?g@@YAXUX@@@Z"(<{ %struct.X, [3 x i8] }>* inalloca) {{.*}} {
// WIN32:   call x86_thiscallcc void @"\01??1X@@QAE@XZ"(%struct.X* {{.*}})
// WIN32: }
void f() {
  g(X());
}
// WIN32: define dso_local void @"\01?f@@YAXXZ"() {{.*}} {
// WIN32-NOT: call {{.*}} @"\01??1X@@QAE@XZ"
// WIN32: }


namespace test2 {
// We used to crash on this due to the mixture of POD byval and non-trivial
// byval.

struct NonTrivial {
  NonTrivial();
  NonTrivial(const NonTrivial &o);
  ~NonTrivial();
  int a;
};
struct POD { int b; };

int foo(NonTrivial a, POD b);
void bar() {
  POD b;
  b.b = 13;
  int c = foo(NonTrivial(), b);
}
// WIN32-LABEL: define dso_local void @"\01?bar@test2@@YAXXZ"() {{.*}} {
// WIN32:   %[[argmem:[^ ]*]] = alloca inalloca [[argmem_ty:<{ %"struct.test2::NonTrivial", %"struct.test2::POD" }>]]
// WIN32:   getelementptr inbounds [[argmem_ty]], [[argmem_ty]]* %[[argmem]], i32 0, i32 1
// WIN32:   call void @llvm.memcpy
// WIN32:   getelementptr inbounds [[argmem_ty]], [[argmem_ty]]* %[[argmem]], i32 0, i32 0
// WIN32:   call x86_thiscallcc %"struct.test2::NonTrivial"* @"\01??0NonTrivial@test2@@QAE@XZ"
// WIN32:   call i32 @"\01?foo@test2@@YAHUNonTrivial@1@UPOD@1@@Z"([[argmem_ty]]* inalloca %argmem)
// WIN32:   ret void
// WIN32: }

}

namespace test3 {

// Check that we padded the inalloca struct to a multiple of 4.
struct NonTrivial {
  NonTrivial();
  NonTrivial(const NonTrivial &o);
  ~NonTrivial();
  int a;
};
void foo(NonTrivial a, bool b) { }
// WIN32-LABEL: define dso_local void @"\01?foo@test3@@YAXUNonTrivial@1@_N@Z"(<{ %"struct.test3::NonTrivial", i8, [3 x i8] }>* inalloca)

}

// We would crash here because the later definition of ForwardDeclare1 results
// in a different IR type for the value we want to store.  However, the alloca's
// type will use the argument type selected by fn1.
struct ForwardDeclare1;

typedef void (*FnPtr1)(ForwardDeclare1);
void fn1(FnPtr1 a, SmallWithDtor b) { }

struct ForwardDeclare1 {};

void fn2(FnPtr1 a, SmallWithDtor b) { fn1(a, b); };
// WIN32-LABEL: define dso_local void @"\01?fn2@@YAXP6AXUForwardDeclare1@@@ZUSmallWithDtor@@@Z"
// WIN32:   %[[a:[^ ]*]] = getelementptr inbounds [[argmem_ty:<{ {}\*, %struct.SmallWithDtor }>]], [[argmem_ty:<{ {}\*, %struct.SmallWithDtor }>]]* %{{.*}}, i32 0, i32 0
// WIN32:   %[[a1:[^ ]*]] = bitcast {}** %[[a]] to void [[dst_ty:\(%struct.ForwardDeclare1\*\)\*]]*
// WIN32:   %[[argmem:[^ ]*]] = alloca inalloca [[argmem_ty]]
// WIN32:   %[[gep1:[^ ]*]] = getelementptr inbounds [[argmem_ty]], [[argmem_ty]]* %[[argmem]], i32 0, i32 1
// WIN32:   %[[bc1:[^ ]*]] = bitcast %struct.SmallWithDtor* %[[gep1]] to i8*
// WIN32:   call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %[[bc1]], i8* align 4 {{.*}}, i32 4, i1 false)
// WIN32:   %[[a2:[^ ]*]] = load void [[dst_ty]], void [[dst_ty]]* %[[a1]], align 4
// WIN32:   %[[gep2:[^ ]*]] = getelementptr inbounds [[argmem_ty]], [[argmem_ty]]* %[[argmem]], i32 0, i32 0
// WIN32:   %[[addr:[^ ]*]] = bitcast {}** %[[gep2]] to void [[dst_ty]]*
// WIN32:   store void [[dst_ty]] %[[a2]], void [[dst_ty]]* %[[addr]], align 4
// WIN32:   call void @"\01?fn1@@YAXP6AXUForwardDeclare1@@@ZUSmallWithDtor@@@Z"([[argmem_ty]]* inalloca %[[argmem]])

namespace pr30293 {
// Virtual methods living in a secondary vtable take i8* as their 'this'
// parameter because the 'this' parameter on entry points to the secondary
// vptr. We used to have a bug where we didn't apply this rule consistently,
// and it would cause assertion failures when used with inalloca.
struct A {
  virtual void f();
};
struct B {
  virtual void __cdecl h(SmallWithDtor);
};
struct C final : A, B {
  void g();
  void __cdecl h(SmallWithDtor);
  void f();
};
void C::g() { return h(SmallWithDtor()); }

// WIN32-LABEL: define dso_local x86_thiscallcc void @"\01?g@C@pr30293@@QAEXXZ"(%"struct.pr30293::C"* %this)
// WIN32: call x86_thiscallcc %struct.SmallWithDtor* @"\01??0SmallWithDtor@@QAE@XZ"
// WIN32: call void @"\01?h@C@pr30293@@UAAXUSmallWithDtor@@@Z"(<{ i8*, %struct.SmallWithDtor }>* inalloca %{{[^,)]*}})
// WIN32: declare dso_local void @"\01?h@C@pr30293@@UAAXUSmallWithDtor@@@Z"(<{ i8*, %struct.SmallWithDtor }>* inalloca)

// WIN64-LABEL: define dso_local void @"\01?g@C@pr30293@@QEAAXXZ"(%"struct.pr30293::C"* %this)
// WIN64: declare dso_local void @"\01?h@C@pr30293@@UEAAXUSmallWithDtor@@@Z"(i8*, i32)
}
