// RUN: %clang_cc1 -triple arm64-none-linux-gnu -emit-llvm -w -o - %s | FileCheck %s

// Check differences between the generic Itanium ABI, the AArch32 version and
// the AArch64 version.

////////////////////////////////////////////////////////////////////////////////

// The ABI says that the key function is the "textually first, non-inline,
// non-pure, virtual member function". The generic version decides this after
// the completion of the class definition; the AArch32 version decides this at
// the end of the translation unit.

// We construct a class which needs a VTable here under generic ABI, but not
// AArch32.

// (see next section for explanation of guard)
// CHECK: @_ZGVZ15guard_variablesiE4mine = internal global i64 0

// CHECK: @_ZTV16CheckKeyFunction =
struct CheckKeyFunction {
  virtual void foo();
};

// This is not inline when CheckKeyFunction is completed, so
// CheckKeyFunction::foo is the key function. VTables should be emitted.
inline void CheckKeyFunction::foo() {
}

////////////////////////////////////////////////////////////////////////////////

// Guard variables only specify and use the low bit to determine status, rather
// than the low byte as in the generic Itanium ABI. However, unlike 32-bit ARM,
// they *are* 64-bits wide so check that in case confusion has occurred.

class Guarded {
public:
  Guarded(int i);
  ~Guarded();
};

void guard_variables(int a) {
  static Guarded mine(a);
// CHECK: [[GUARDBIT:%[0-9]+]] = and i8 {{%[0-9]+}}, 1
// CHECK: icmp eq i8 [[GUARDBIT]], 0

  // As guards are 64-bit, these helpers should take 64-bit pointers.
// CHECK: call i32 @__cxa_guard_acquire(ptr
// CHECK: call void @__cxa_guard_release(ptr
}

////////////////////////////////////////////////////////////////////////////////

// Member function pointers use the adj field to distinguish between virtual and
// nonvirtual members. As a result the adjustment is shifted (if ptr was used, a
// mask would be expected instead).

class C {
  int a();
  virtual int b();
};


int member_pointer(C &c, int (C::*func)()) {
// CHECK: ashr i64 %[[MEMPTRADJ:[0-9a-z.]+]], 1
// CHECK: %[[ISVIRTUAL:[0-9]+]] = and i64 %[[MEMPTRADJ]], 1
// CHECK: icmp ne i64 %[[ISVIRTUAL]], 0
  return (c.*func)();
}

////////////////////////////////////////////////////////////////////////////////

// AArch64 PCS says that va_list type is based on "struct __va_list ..." in the
// std namespace, which means it should mangle as "St9__va_list".

// CHECK: @_Z7va_funcSt9__va_list
void va_func(__builtin_va_list l) {
}

////////////////////////////////////////////////////////////////////////////////

// AArch64 constructors (like generic Itanium, but unlike AArch32) do not return
// "this".

void test_constructor() {
  Guarded g(42);
// CHECK: call void @_ZN7GuardedC1Ei
}

////////////////////////////////////////////////////////////////////////////////

// In principle the AArch32 ABI allows this to be accomplished via a call to
// __aeabi_atexit instead of __cxa_atexit. Clang doesn't make use of this at the
// moment, but it's definitely not allowed for AArch64.

// CHECK: call i32 @__cxa_atexit
Guarded g(42);
