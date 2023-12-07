// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=i386-pc-linux | FileCheck -check-prefix LINUX %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=i386-pc-win32 -mconstructor-aliases -fno-rtti | FileCheck -check-prefix WIN32 --check-prefix WIN %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=thumb-pc-win32 -mconstructor-aliases -fno-rtti | FileCheck -check-prefix WOA --check-prefix WIN %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=x86_64-pc-win32 -mconstructor-aliases -fno-rtti | FileCheck -check-prefix WIN64 --check-prefix WIN %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=aarch64-windows-msvc -mconstructor-aliases -fno-rtti | FileCheck -check-prefix WOA64 --check-prefix WIN %s

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

struct SmallWithPrivate {
private:
 int i;
};

struct SmallWithSmallWithPrivate {
  SmallWithPrivate p;
};

// WIN32: declare dso_local void @"{{.*take_bools_and_chars.*}}"
// WIN32:       (ptr inalloca(<{ i8, [3 x i8], i8, [3 x i8], %struct.SmallWithDtor, i8, [3 x i8], i8, [3 x i8], i32, i8, [3 x i8] }>)
void take_bools_and_chars(char a, char b, SmallWithDtor c, char d, bool e, int f, bool g);
void call_bools_and_chars() {
  take_bools_and_chars('A', 'B', SmallWithDtor(), 'D', true, 13, false);
}

// Returning structs that fit into a register.
Small small_return() { return Small(); }
// LINUX-LABEL: define{{.*}} void @_Z12small_returnv(ptr noalias sret(%struct.Small) align 4 %agg.result)
// WIN32: define dso_local i32 @"?small_return@@YA?AUSmall@@XZ"()
// WIN64: define dso_local i32 @"?small_return@@YA?AUSmall@@XZ"()
// WOA64: define dso_local i32 @"?small_return@@YA?AUSmall@@XZ"()

Medium medium_return() { return Medium(); }
// LINUX-LABEL: define{{.*}} void @_Z13medium_returnv(ptr noalias sret(%struct.Medium) align 4 %agg.result)
// WIN32: define dso_local i64 @"?medium_return@@YA?AUMedium@@XZ"()
// WIN64: define dso_local i64 @"?medium_return@@YA?AUMedium@@XZ"()
// WOA64: define dso_local i64 @"?medium_return@@YA?AUMedium@@XZ"()

// Returning structs that fit into a register but are not POD.
SmallCpp11NotCpp03Pod small_non_pod_return() { return SmallCpp11NotCpp03Pod(); }
// LINUX-LABEL: define{{.*}} void @_Z20small_non_pod_returnv(ptr noalias sret(%struct.SmallCpp11NotCpp03Pod) align 4 %agg.result)
// WIN32: define dso_local void @"?small_non_pod_return@@YA?AUSmallCpp11NotCpp03Pod@@XZ"(ptr noalias sret(%struct.SmallCpp11NotCpp03Pod) align 4 %agg.result)
// WIN64: define dso_local void @"?small_non_pod_return@@YA?AUSmallCpp11NotCpp03Pod@@XZ"(ptr noalias sret(%struct.SmallCpp11NotCpp03Pod) align 4 %agg.result)
// WOA64: define dso_local void @"?small_non_pod_return@@YA?AUSmallCpp11NotCpp03Pod@@XZ"(ptr inreg noalias sret(%struct.SmallCpp11NotCpp03Pod) align 4 %agg.result)

SmallWithCtor small_with_ctor_return() { return SmallWithCtor(); }
// LINUX-LABEL: define{{.*}} void @_Z22small_with_ctor_returnv(ptr noalias sret(%struct.SmallWithCtor) align 4 %agg.result)
// WIN32: define dso_local void @"?small_with_ctor_return@@YA?AUSmallWithCtor@@XZ"(ptr noalias sret(%struct.SmallWithCtor) align 4 %agg.result)
// WIN64: define dso_local void @"?small_with_ctor_return@@YA?AUSmallWithCtor@@XZ"(ptr noalias sret(%struct.SmallWithCtor) align 4 %agg.result)
// FIXME: The 'sret' mark here doesn't seem to be enough to convince LLVM to
// preserve the hidden sret pointer in R0 across the function.
// WOA: define dso_local arm_aapcs_vfpcc void @"?small_with_ctor_return@@YA?AUSmallWithCtor@@XZ"(ptr noalias sret(%struct.SmallWithCtor) align 4 %agg.result)
// WOA64: define dso_local void @"?small_with_ctor_return@@YA?AUSmallWithCtor@@XZ"(ptr inreg noalias sret(%struct.SmallWithCtor) align 4 %agg.result)

SmallWithDtor small_with_dtor_return() { return SmallWithDtor(); }
// LINUX-LABEL: define{{.*}} void @_Z22small_with_dtor_returnv(ptr noalias sret(%struct.SmallWithDtor) align 4 %agg.result)
// WIN32: define dso_local void @"?small_with_dtor_return@@YA?AUSmallWithDtor@@XZ"(ptr noalias sret(%struct.SmallWithDtor) align 4 %agg.result)
// WIN64: define dso_local void @"?small_with_dtor_return@@YA?AUSmallWithDtor@@XZ"(ptr noalias sret(%struct.SmallWithDtor) align 4 %agg.result)
// WOA64: define dso_local void @"?small_with_dtor_return@@YA?AUSmallWithDtor@@XZ"(ptr inreg noalias sret(%struct.SmallWithDtor) align 4 %agg.result)

SmallWithVftable small_with_vftable_return() { return SmallWithVftable(); }
// LINUX-LABEL: define{{.*}} void @_Z25small_with_vftable_returnv(ptr noalias sret(%struct.SmallWithVftable) align 4 %agg.result)
// WIN32: define dso_local void @"?small_with_vftable_return@@YA?AUSmallWithVftable@@XZ"(ptr noalias sret(%struct.SmallWithVftable) align 4 %agg.result)
// WIN64: define dso_local void @"?small_with_vftable_return@@YA?AUSmallWithVftable@@XZ"(ptr noalias sret(%struct.SmallWithVftable) align 8 %agg.result)
// WOA64: define dso_local void @"?small_with_vftable_return@@YA?AUSmallWithVftable@@XZ"(ptr inreg noalias sret(%struct.SmallWithVftable) align 8 %agg.result)

MediumWithCopyCtor medium_with_copy_ctor_return() { return MediumWithCopyCtor(); }
// LINUX-LABEL: define{{.*}} void @_Z28medium_with_copy_ctor_returnv(ptr noalias sret(%struct.MediumWithCopyCtor) align 4 %agg.result)
// WIN32: define dso_local void @"?medium_with_copy_ctor_return@@YA?AUMediumWithCopyCtor@@XZ"(ptr noalias sret(%struct.MediumWithCopyCtor) align 4 %agg.result)
// WIN64: define dso_local void @"?medium_with_copy_ctor_return@@YA?AUMediumWithCopyCtor@@XZ"(ptr noalias sret(%struct.MediumWithCopyCtor) align 4 %agg.result)
// WOA: define dso_local arm_aapcs_vfpcc void @"?medium_with_copy_ctor_return@@YA?AUMediumWithCopyCtor@@XZ"(ptr noalias sret(%struct.MediumWithCopyCtor) align 4 %agg.result)
// WOA64: define dso_local void @"?medium_with_copy_ctor_return@@YA?AUMediumWithCopyCtor@@XZ"(ptr inreg noalias sret(%struct.MediumWithCopyCtor) align 4 %agg.result)

// Returning a large struct that doesn't fit into a register.
Big big_return() { return Big(); }
// LINUX-LABEL: define{{.*}} void @_Z10big_returnv(ptr noalias sret(%struct.Big) align 4 %agg.result)
// WIN32: define dso_local void @"?big_return@@YA?AUBig@@XZ"(ptr noalias sret(%struct.Big) align 4 %agg.result)
// WIN64: define dso_local void @"?big_return@@YA?AUBig@@XZ"(ptr noalias sret(%struct.Big) align 4 %agg.result)
// WOA64: define dso_local void @"?big_return@@YA?AUBig@@XZ"(ptr noalias sret(%struct.Big) align 4 %agg.result)


void small_arg(Small s) {}
// LINUX-LABEL: define{{.*}} void @_Z9small_arg5Small(i32 %s.0)
// WIN32: define dso_local void @"?small_arg@@YAXUSmall@@@Z"(i32 %s.0)
// WIN64: define dso_local void @"?small_arg@@YAXUSmall@@@Z"(i32 %s.coerce)
// WOA: define dso_local arm_aapcs_vfpcc void @"?small_arg@@YAXUSmall@@@Z"([1 x i32] %s.coerce)

void medium_arg(Medium s) {}
// LINUX-LABEL: define{{.*}} void @_Z10medium_arg6Medium(i32 %s.0, i32 %s.1)
// WIN32: define dso_local void @"?medium_arg@@YAXUMedium@@@Z"(i32 %s.0, i32 %s.1)
// WIN64: define dso_local void @"?medium_arg@@YAXUMedium@@@Z"(i64 %s.coerce)
// WOA: define dso_local arm_aapcs_vfpcc void @"?medium_arg@@YAXUMedium@@@Z"([2 x i32] %s.coerce)

void base_no_byval_arg(BaseNoByval s) {}
// LINUX-LABEL: define{{.*}} void @_Z17base_no_byval_arg11BaseNoByval(ptr noundef byval(%struct.BaseNoByval) align 4 %s)
// WIN32: define dso_local void @"?base_no_byval_arg@@YAXUBaseNoByval@@@Z"(i32 %s.0, i32 %s.1)
// WIN64: define dso_local void @"?base_no_byval_arg@@YAXUBaseNoByval@@@Z"(i64 %s.coerce)
// WOA: define dso_local arm_aapcs_vfpcc void @"?base_no_byval_arg@@YAXUBaseNoByval@@@Z"([2 x i32] %s.coerce)

void small_arg_with_ctor(SmallWithCtor s) {}
// LINUX-LABEL: define{{.*}} void @_Z19small_arg_with_ctor13SmallWithCtor(ptr noundef byval(%struct.SmallWithCtor) align 4 %s)
// WIN32: define dso_local void @"?small_arg_with_ctor@@YAXUSmallWithCtor@@@Z"(i32 %s.0)
// WIN64: define dso_local void @"?small_arg_with_ctor@@YAXUSmallWithCtor@@@Z"(i32 %s.coerce)
// WOA: define dso_local arm_aapcs_vfpcc void @"?small_arg_with_ctor@@YAXUSmallWithCtor@@@Z"([1 x i32] %s.coerce)

// FIXME: We could coerce to a series of i32s here if we wanted to.
void multibyte_arg(Multibyte s) {}
// LINUX-LABEL: define{{.*}} void @_Z13multibyte_arg9Multibyte(ptr noundef byval(%struct.Multibyte) align 4 %s)
// WIN32: define dso_local void @"?multibyte_arg@@YAXUMultibyte@@@Z"(ptr noundef byval(%struct.Multibyte) align 4 %s)
// WIN64: define dso_local void @"?multibyte_arg@@YAXUMultibyte@@@Z"(i32 %s.coerce)
// WOA: define dso_local arm_aapcs_vfpcc void @"?multibyte_arg@@YAXUMultibyte@@@Z"([1 x i32] %s.coerce)

void packed_arg(Packed s) {}
// LINUX-LABEL: define{{.*}} void @_Z10packed_arg6Packed(ptr noundef byval(%struct.Packed) align 4 %s)
// WIN32: define dso_local void @"?packed_arg@@YAXUPacked@@@Z"(ptr noundef byval(%struct.Packed) align 4 %s)
// WIN64: define dso_local void @"?packed_arg@@YAXUPacked@@@Z"(ptr noundef %s)

// Test that dtors are invoked in the callee.
void small_arg_with_dtor(SmallWithDtor s) {}
// WIN32: define dso_local void @"?small_arg_with_dtor@@YAXUSmallWithDtor@@@Z"(ptr inalloca(<{ %struct.SmallWithDtor }>) %0) {{.*}} {
// WIN32:   call x86_thiscallcc void @"??1SmallWithDtor@@QAE@XZ"
// WIN32: }
// WIN64: define dso_local void @"?small_arg_with_dtor@@YAXUSmallWithDtor@@@Z"(i32 %s.coerce) {{.*}} {
// WIN64:   call void @"??1SmallWithDtor@@QEAA@XZ"
// WIN64: }
// WOA64: define dso_local void @"?small_arg_with_dtor@@YAXUSmallWithDtor@@@Z"(i64 %s.coerce) {{.*}} {
// WOA64:   call void @"??1SmallWithDtor@@QEAA@XZ"(ptr {{[^,]*}} %s)
// WOA64: }

// FIXME: MSVC incompatible!
// WOA: define dso_local arm_aapcs_vfpcc void @"?small_arg_with_dtor@@YAXUSmallWithDtor@@@Z"(ptr noundef %s) {{.*}} {
// WOA:   call arm_aapcs_vfpcc void @"??1SmallWithDtor@@QAA@XZ"(ptr {{[^,]*}} %s)
// WOA: }


// Test that the eligible non-aggregate is passed directly, but returned
// indirectly on ARM64 Windows.
// WOA64: define dso_local void @"?small_arg_with_private_member@@YA?AUSmallWithPrivate@@U1@@Z"(ptr inreg noalias sret(%struct.SmallWithPrivate) align 4 %agg.result, i64 %s.coerce) {{.*}} {
SmallWithPrivate small_arg_with_private_member(SmallWithPrivate s) { return s; }

// WOA64: define dso_local i32 @"?small_arg_with_small_struct_with_private_member@@YA?AUSmallWithSmallWithPrivate@@U1@@Z"(i64 %s.coerce) {{.*}} {
// WIN64: define dso_local i32 @"?small_arg_with_small_struct_with_private_member@@YA?AUSmallWithSmallWithPrivate@@U1@@Z"(i32 %s.coerce) {{.*}} {
SmallWithSmallWithPrivate small_arg_with_small_struct_with_private_member(SmallWithSmallWithPrivate s) { return s; }

void call_small_arg_with_dtor() {
  small_arg_with_dtor(SmallWithDtor());
}
// WIN64-LABEL: define dso_local void @"?call_small_arg_with_dtor@@YAXXZ"()
// WIN64:   call noundef ptr @"??0SmallWithDtor@@QEAA@XZ"
// WIN64:   call void @"?small_arg_with_dtor@@YAXUSmallWithDtor@@@Z"(i32 %{{.*}})
// WIN64:   ret void

// Test that references aren't destroyed in the callee.
void ref_small_arg_with_dtor(const SmallWithDtor &s) { }
// WIN32: define dso_local void @"?ref_small_arg_with_dtor@@YAXABUSmallWithDtor@@@Z"(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %s) {{.*}} {
// WIN32-NOT:   call x86_thiscallcc void @"??1SmallWithDtor@@QAE@XZ"
// WIN32: }
// WIN64-LABEL: define dso_local void @"?ref_small_arg_with_dtor@@YAXAEBUSmallWithDtor@@@Z"(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %s)

void big_arg_with_dtor(BigWithDtor s) {}
// WIN64-LABEL: define dso_local void @"?big_arg_with_dtor@@YAXUBigWithDtor@@@Z"(ptr noundef %s)
// WIN64:   call void @"??1BigWithDtor@@QEAA@XZ"
// WIN64: }

void call_big_arg_with_dtor() {
  big_arg_with_dtor(BigWithDtor());
}
// We can elide the copy of the temporary in the caller, because this object is
// larger than 8 bytes and is passed indirectly.
// WIN64-LABEL: define dso_local void @"?call_big_arg_with_dtor@@YAXXZ"()
// WIN64:   call noundef ptr @"??0BigWithDtor@@QEAA@XZ"
// WIN64:   call void @"?big_arg_with_dtor@@YAXUBigWithDtor@@@Z"(ptr noundef %{{.*}})
// WIN64-NOT: call void @"??1BigWithDtor@@QEAA@XZ"
// WIN64:   ret void

// Test that temporaries passed by reference are destroyed in the caller.
void temporary_ref_with_dtor() {
  ref_small_arg_with_dtor(SmallWithDtor());
}
// WIN32: define dso_local void @"?temporary_ref_with_dtor@@YAXXZ"() {{.*}} {
// WIN32:   call x86_thiscallcc noundef ptr @"??0SmallWithDtor@@QAE@XZ"
// WIN32:   call void @"?ref_small_arg_with_dtor@@YAXABUSmallWithDtor@@@Z"
// WIN32:   call x86_thiscallcc void @"??1SmallWithDtor@@QAE@XZ"
// WIN32: }

void takes_two_by_val_with_dtor(SmallWithDtor a, SmallWithDtor b);
void eh_cleanup_arg_with_dtor() {
  takes_two_by_val_with_dtor(SmallWithDtor(), SmallWithDtor());
}
//   When exceptions are off, we don't have any cleanups.  See
//   microsoft-abi-exceptions.cpp for these cleanups.
// WIN32: define dso_local void @"?eh_cleanup_arg_with_dtor@@YAXXZ"() {{.*}} {
// WIN32:   call x86_thiscallcc noundef ptr @"??0SmallWithDtor@@QAE@XZ"
// WIN32:   call x86_thiscallcc noundef ptr @"??0SmallWithDtor@@QAE@XZ"
// WIN32:   call void @"?takes_two_by_val_with_dtor@@YAXUSmallWithDtor@@0@Z"
// WIN32-NOT: call x86_thiscallcc void @"??1SmallWithDtor@@QAE@XZ"
// WIN32: }

void small_arg_with_vftable(SmallWithVftable s) {}
// LINUX-LABEL: define{{.*}} void @_Z22small_arg_with_vftable16SmallWithVftable(ptr noundef %s)
// WIN32: define dso_local void @"?small_arg_with_vftable@@YAXUSmallWithVftable@@@Z"(ptr inalloca(<{ %struct.SmallWithVftable }>) %0)
// WIN64: define dso_local void @"?small_arg_with_vftable@@YAXUSmallWithVftable@@@Z"(ptr noundef %s)
// WOA64: define dso_local void @"?small_arg_with_vftable@@YAXUSmallWithVftable@@@Z"(ptr noundef %s)

void medium_arg_with_copy_ctor(MediumWithCopyCtor s) {}
// LINUX-LABEL: define{{.*}} void @_Z25medium_arg_with_copy_ctor18MediumWithCopyCtor(ptr noundef %s)
// WIN32: define dso_local void @"?medium_arg_with_copy_ctor@@YAXUMediumWithCopyCtor@@@Z"(ptr inalloca(<{ %struct.MediumWithCopyCtor }>) %0)
// WIN64: define dso_local void @"?medium_arg_with_copy_ctor@@YAXUMediumWithCopyCtor@@@Z"(ptr noundef %s)
// WOA: define dso_local arm_aapcs_vfpcc void @"?medium_arg_with_copy_ctor@@YAXUMediumWithCopyCtor@@@Z"(ptr noundef %s)
// WOA64: define dso_local void @"?medium_arg_with_copy_ctor@@YAXUMediumWithCopyCtor@@@Z"(ptr noundef %s)

void big_arg(Big s) {}
// LINUX-LABEL: define{{.*}} void @_Z7big_arg3Big(ptr noundef byval(%struct.Big) align 4 %s)
// WIN32: define dso_local void @"?big_arg@@YAXUBig@@@Z"(ptr noundef byval(%struct.Big) align 4 %s)
// WIN64: define dso_local void @"?big_arg@@YAXUBig@@@Z"(ptr noundef %s)

// PR27607: We would attempt to load i32 value out of the reference instead of
// just loading the pointer from the struct during argument expansion.
struct RefField {
  RefField(int &x);
  int &x;
};
void takes_ref_field(RefField s) {}
// LINUX-LABEL: define{{.*}} void @_Z15takes_ref_field8RefField(ptr noundef byval(%struct.RefField) align 4 %s)
// WIN32: define dso_local void @"?takes_ref_field@@YAXURefField@@@Z"(ptr %s.0)
// WIN64: define dso_local void @"?takes_ref_field@@YAXURefField@@@Z"(i64 %s.coerce)

void pass_ref_field() {
  int x;
  takes_ref_field(RefField(x));
}
// LINUX-LABEL: define{{.*}} void @_Z14pass_ref_fieldv()
// LINUX: call void @_Z15takes_ref_field8RefField(ptr noundef byval(%struct.RefField) align 4 %{{.*}})
// WIN32-LABEL: define dso_local void @"?pass_ref_field@@YAXXZ"()
// WIN32: call void @"?takes_ref_field@@YAXURefField@@@Z"(ptr %{{.*}})
// WIN64-LABEL: define dso_local void @"?pass_ref_field@@YAXXZ"()
// WIN64: call void @"?takes_ref_field@@YAXURefField@@@Z"(i64 %{{.*}})

class Class {
 public:
  Small thiscall_method_small() { return Small(); }
  // LINUX: define {{.*}} void @_ZN5Class21thiscall_method_smallEv(ptr noalias sret(%struct.Small) align 4 %agg.result, ptr {{[^,]*}} %this)
  // WIN32: define {{.*}} x86_thiscallcc void @"?thiscall_method_small@Class@@QAE?AUSmall@@XZ"(ptr {{[^,]*}} %this, ptr noalias sret(%struct.Small) align 4 %agg.result)
  // WIN64: define linkonce_odr dso_local void @"?thiscall_method_small@Class@@QEAA?AUSmall@@XZ"(ptr {{[^,]*}} %this, ptr noalias sret(%struct.Small) align 4 %agg.result)
  // WOA64: define linkonce_odr dso_local void @"?thiscall_method_small@Class@@QEAA?AUSmall@@XZ"(ptr {{[^,]*}} %this, ptr inreg noalias sret(%struct.Small) align 4 %agg.result)

  SmallWithCtor thiscall_method_small_with_ctor() { return SmallWithCtor(); }
  // LINUX: define {{.*}} void @_ZN5Class31thiscall_method_small_with_ctorEv(ptr noalias sret(%struct.SmallWithCtor) align 4 %agg.result, ptr {{[^,]*}} %this)
  // WIN32: define {{.*}} x86_thiscallcc void @"?thiscall_method_small_with_ctor@Class@@QAE?AUSmallWithCtor@@XZ"(ptr {{[^,]*}} %this, ptr noalias sret(%struct.SmallWithCtor) align 4 %agg.result)
  // WIN64: define linkonce_odr dso_local void @"?thiscall_method_small_with_ctor@Class@@QEAA?AUSmallWithCtor@@XZ"(ptr {{[^,]*}} %this, ptr noalias sret(%struct.SmallWithCtor) align 4 %agg.result)
  // WOA64: define linkonce_odr dso_local void @"?thiscall_method_small_with_ctor@Class@@QEAA?AUSmallWithCtor@@XZ"(ptr {{[^,]*}} %this, ptr inreg noalias sret(%struct.SmallWithCtor) align 4 %agg.result)

  Small __cdecl cdecl_method_small() { return Small(); }
  // LINUX: define {{.*}} void @_ZN5Class18cdecl_method_smallEv(ptr noalias sret(%struct.Small) align 4 %agg.result, ptr {{[^,]*}} %this)
  // WIN32: define {{.*}} void @"?cdecl_method_small@Class@@QAA?AUSmall@@XZ"(ptr {{[^,]*}} %this, ptr noalias sret(%struct.Small) align 4 %agg.result)
  // WIN64: define linkonce_odr dso_local void @"?cdecl_method_small@Class@@QEAA?AUSmall@@XZ"(ptr {{[^,]*}} %this, ptr noalias sret(%struct.Small) align 4 %agg.result)

  Big __cdecl cdecl_method_big() { return Big(); }
  // LINUX: define {{.*}} void @_ZN5Class16cdecl_method_bigEv(ptr noalias sret(%struct.Big) align 4 %agg.result, ptr {{[^,]*}} %this)
  // WIN32: define {{.*}} void @"?cdecl_method_big@Class@@QAA?AUBig@@XZ"(ptr {{[^,]*}} %this, ptr noalias sret(%struct.Big) align 4 %agg.result)
  // WIN64: define linkonce_odr dso_local void @"?cdecl_method_big@Class@@QEAA?AUBig@@XZ"(ptr {{[^,]*}} %this, ptr noalias sret(%struct.Big) align 4 %agg.result)
  // WOA64: define linkonce_odr dso_local void @"?cdecl_method_big@Class@@QEAA?AUBig@@XZ"(ptr {{[^,]*}} %this, ptr inreg noalias sret(%struct.Big) align 4 %agg.result)

  void thiscall_method_arg(Empty s) {}
  // LINUX: define {{.*}} void @_ZN5Class19thiscall_method_argE5Empty(ptr {{[^,]*}} %this)
  // WIN32: define {{.*}} void @"?thiscall_method_arg@Class@@QAEXUEmpty@@@Z"(ptr {{[^,]*}} %this, ptr noundef byval(%struct.Empty) align 4 %s)
  // WIN64: define linkonce_odr dso_local void @"?thiscall_method_arg@Class@@QEAAXUEmpty@@@Z"(ptr {{[^,]*}} %this, i8 %s.coerce)

  void thiscall_method_arg(EmptyWithCtor s) {}
  // LINUX: define {{.*}} void @_ZN5Class19thiscall_method_argE13EmptyWithCtor(ptr {{[^,]*}} %this)
  // WIN32: define {{.*}} void @"?thiscall_method_arg@Class@@QAEXUEmptyWithCtor@@@Z"(ptr {{[^,]*}} %this, ptr noundef byval(%struct.EmptyWithCtor) align 4 %s)
  // WIN64: define linkonce_odr dso_local void @"?thiscall_method_arg@Class@@QEAAXUEmptyWithCtor@@@Z"(ptr {{[^,]*}} %this, i8 %s.coerce)

  void thiscall_method_arg(Small s) {}
  // LINUX: define {{.*}} void @_ZN5Class19thiscall_method_argE5Small(ptr {{[^,]*}} %this, i32 %s.0)
  // WIN32: define {{.*}} void @"?thiscall_method_arg@Class@@QAEXUSmall@@@Z"(ptr {{[^,]*}} %this, i32 %s.0)
  // WIN64: define linkonce_odr dso_local void @"?thiscall_method_arg@Class@@QEAAXUSmall@@@Z"(ptr {{[^,]*}} %this, i32 %s.coerce)

  void thiscall_method_arg(SmallWithCtor s) {}
  // LINUX: define {{.*}} void @_ZN5Class19thiscall_method_argE13SmallWithCtor(ptr {{[^,]*}} %this, ptr noundef byval(%struct.SmallWithCtor) align 4 %s)
  // WIN32: define {{.*}} void @"?thiscall_method_arg@Class@@QAEXUSmallWithCtor@@@Z"(ptr {{[^,]*}} %this, i32 %s.0)
  // WIN64: define linkonce_odr dso_local void @"?thiscall_method_arg@Class@@QEAAXUSmallWithCtor@@@Z"(ptr {{[^,]*}} %this, i32 %s.coerce)

  void thiscall_method_arg(Big s) {}
  // LINUX: define {{.*}} void @_ZN5Class19thiscall_method_argE3Big(ptr {{[^,]*}} %this, ptr noundef byval(%struct.Big) align 4 %s)
  // WIN32: define {{.*}} void @"?thiscall_method_arg@Class@@QAEXUBig@@@Z"(ptr {{[^,]*}} %this, ptr noundef byval(%struct.Big) align 4 %s)
  // WIN64: define linkonce_odr dso_local void @"?thiscall_method_arg@Class@@QEAAXUBig@@@Z"(ptr {{[^,]*}} %this, ptr noundef %s)
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
// WIN32: define dso_local void @"?g@@YAXUX@@@Z"(ptr inalloca(<{ %struct.X, [3 x i8] }>) %0) {{.*}} {
// WIN32:   call x86_thiscallcc void @"??1X@@QAE@XZ"(ptr {{.*}})
// WIN32: }
void f() {
  g(X());
}
// WIN32: define dso_local void @"?f@@YAXXZ"() {{.*}} {
// WIN32-NOT: call {{.*}} @"??1X@@QAE@XZ"
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
// WIN32-LABEL: define dso_local void @"?bar@test2@@YAXXZ"() {{.*}} {
// WIN32:   %[[argmem:[^ ]*]] = alloca inalloca [[argmem_ty:<{ %"struct.test2::NonTrivial", %"struct.test2::POD" }>]]
// WIN32:   getelementptr inbounds [[argmem_ty]], ptr %[[argmem]], i32 0, i32 1
// WIN32:   call void @llvm.memcpy
// WIN32:   getelementptr inbounds [[argmem_ty]], ptr %[[argmem]], i32 0, i32 0
// WIN32:   call x86_thiscallcc noundef ptr @"??0NonTrivial@test2@@QAE@XZ"
// WIN32:   call noundef i32 @"?foo@test2@@YAHUNonTrivial@1@UPOD@1@@Z"(ptr inalloca([[argmem_ty]]) %argmem)
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
// WIN32-LABEL: define dso_local void @"?foo@test3@@YAXUNonTrivial@1@_N@Z"(ptr inalloca(<{ %"struct.test3::NonTrivial", i8, [3 x i8] }>) %0)

}

// We would crash here because the later definition of ForwardDeclare1 results
// in a different IR type for the value we want to store.  However, the alloca's
// type will use the argument type selected by fn1.
struct ForwardDeclare1;

typedef void (*FnPtr1)(ForwardDeclare1);
void fn1(FnPtr1 a, SmallWithDtor b) { }

struct ForwardDeclare1 {};

void fn2(FnPtr1 a, SmallWithDtor b) { fn1(a, b); };
// WIN32-LABEL: define dso_local void @"?fn2@@YAXP6AXUForwardDeclare1@@@ZUSmallWithDtor@@@Z"
// WIN32:   %[[a:[^ ]*]] = getelementptr inbounds [[argmem_ty:<{ ptr, %struct.SmallWithDtor }>]], ptr %{{.*}}, i32 0, i32 0
// WIN32:   %[[argmem:[^ ]*]] = alloca inalloca [[argmem_ty:<{ ptr, %struct.SmallWithDtor }>]]
// WIN32:   %[[gep1:[^ ]*]] = getelementptr inbounds [[argmem_ty]], ptr %[[argmem]], i32 0, i32 1
// WIN32:   call void @llvm.memcpy.p0.p0.i32(ptr align 4 %[[gep1]], ptr align 4 {{.*}}, i32 4, i1 false)
// WIN32:   %[[a2:[^ ]*]] = load ptr, ptr %[[a]], align 4
// WIN32:   %[[gep2:[^ ]*]] = getelementptr inbounds [[argmem_ty]], ptr %[[argmem]], i32 0, i32 0
// WIN32:   store ptr %[[a2]], ptr %[[gep2]], align 4
// WIN32:   call void @"?fn1@@YAXP6AXUForwardDeclare1@@@ZUSmallWithDtor@@@Z"(ptr inalloca([[argmem_ty]]) %[[argmem]])

namespace pr30293 {
// Virtual methods living in a secondary vtable take ptr as their 'this'
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

// WIN32-LABEL: define dso_local x86_thiscallcc void @"?g@C@pr30293@@QAEXXZ"(ptr {{[^,]*}} %this)
// WIN32: call x86_thiscallcc noundef ptr @"??0SmallWithDtor@@QAE@XZ"
// WIN32: call void @"?h@C@pr30293@@UAAXUSmallWithDtor@@@Z"(ptr inalloca(<{ ptr, %struct.SmallWithDtor }>) %{{[^,)]*}})
// WIN32: declare dso_local void @"?h@C@pr30293@@UAAXUSmallWithDtor@@@Z"(ptr inalloca(<{ ptr, %struct.SmallWithDtor }>))

// WIN64-LABEL: define dso_local void @"?g@C@pr30293@@QEAAXXZ"(ptr {{[^,]*}} %this)
// WIN64: declare dso_local void @"?h@C@pr30293@@UEAAXUSmallWithDtor@@@Z"(ptr noundef, i32)
}

namespace protected_member_of_member {
struct field { protected: int i; };
struct t1 {
  field f;
};
extern const t1& v1;
t1 f1() { return v1; }
// WIN: define dso_local {{.*}}i32 @"?f1@protected_member_of_member@@YA?AUt1@1@XZ"()
}

namespace default_member_initializer {
struct t1 {
  int i = 3;
};
extern const t1& v1;
t1 f1() { return v1; }
// WIN: define dso_local {{.*}}i32 @"?f1@default_member_initializer@@YA?AUt1@1@XZ"()
}

namespace defaulted_copy_ctor {
struct t1 {
  int i;
  t1(const t1&) = default;
};
extern const t1& v1;
t1 f1() { return v1; }
// WIN: define dso_local {{.*}}i32 @"?f1@defaulted_copy_ctor@@YA?AUt1@1@XZ"()
}
