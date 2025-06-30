// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=cfi-icall -o - %s | FileCheck %s

#define CFI_UNCHECKED_CALLEE __attribute__((cfi_unchecked_callee))

void unchecked(void) CFI_UNCHECKED_CALLEE {}

/// All references to unchecked function with `cfi_unchecked_callee` should have the `cfi_unchecked_callee` wrapper.
// CHECK: @checked = global ptr no_cfi @_Z9uncheckedv
void (*checked)(void) = unchecked;

// CHECK: @unchecked2 = global ptr no_cfi @_Z9uncheckedv
void (CFI_UNCHECKED_CALLEE *unchecked2)(void) = unchecked;

// CHECK: @checked2 = global ptr no_cfi @_Z9uncheckedv
constexpr void (CFI_UNCHECKED_CALLEE *unchecked_constexpr)(void) = unchecked;
void (*checked2)(void) = unchecked_constexpr;

/// Note we still reference the `no_cfi` function rather than the jump table entry.
/// The explicit cast will only silence the warning.
// CHECK: @checked_explicit_cast = global ptr no_cfi @_Z9uncheckedv
void (*checked_explicit_cast)(void) = (void (*)(void))unchecked;

// CHECK: @checked_array = global [3 x ptr] [ptr no_cfi @_Z9uncheckedv, ptr no_cfi @_Z9uncheckedv, ptr no_cfi @_Z9uncheckedv]
void (*checked_array[])(void) = {
  unchecked,
  (void (*)(void))unchecked,
  reinterpret_cast<void (*)(void)>(unchecked),
};

void func_accepting_checked(void (*p)(void)) {}

// CHECK-LABEL: _Z9InvokeCFIv
void InvokeCFI() {
  // CHECK: %0 = load ptr, ptr @checked, align 8
  // CHECK: %1 = call i1 @llvm.type.test(ptr %0, metadata !"_ZTSFvvE")
  checked();
}

// CHECK-LABEL: _Z11InvokeNoCFIv
void InvokeNoCFI() {
  // CHECK:  %0 = load ptr, ptr @unchecked2, align 8
  // CHECK:  call void %0()
  unchecked2();
}

struct A {
  void unchecked_method() CFI_UNCHECKED_CALLEE {}
  virtual void unchecked_virtual_method() CFI_UNCHECKED_CALLEE {}
  static void unchecked_static_method() CFI_UNCHECKED_CALLEE {}
  int unchecked_const_method() const CFI_UNCHECKED_CALLEE { return 0; }
  int unchecked_const_method_int_arg(int n) const CFI_UNCHECKED_CALLEE { return 0; }
};

void h(void) {
  // CHECK: store ptr no_cfi @_Z9uncheckedv, ptr %unchecked_local
  void (*unchecked_local)(void) = unchecked;

  // CHECK: call void @_Z22func_accepting_checkedPFvvE(ptr noundef no_cfi @_Z9uncheckedv)
  func_accepting_checked(unchecked);

  // CHECK:      [[B:%.*]] = load ptr, ptr @checked
  // CHECK-NEXT: call void @_Z22func_accepting_checkedPFvvE(ptr noundef [[B]]) 
  func_accepting_checked(checked);

  // CHECK: store { i64, i64 } { i64 ptrtoint (ptr no_cfi @_ZN1A16unchecked_methodEv to i64), i64 0 }, ptr %A1
  auto A1 = &A::unchecked_method;
  /// Storing unchecked virtual function pointer stores an offset instead. This is part of the
  /// normal Itanium C++ ABI, but let's make sure we don't change anything.
  // CHECK: store { i64, i64 } { i64 1, i64 0 }, ptr %A2
  auto A2 = &A::unchecked_virtual_method;
  // CHECK: store ptr no_cfi @_ZN1A23unchecked_static_methodEv, ptr %A3
  auto A3 = &A::unchecked_static_method;
  // CHECK: store { i64, i64 } { i64 ptrtoint (ptr no_cfi @_ZNK1A22unchecked_const_methodEv to i64), i64 0 }, ptr %A4
  auto A4 = (int(CFI_UNCHECKED_CALLEE A::*)() const)(&A::unchecked_const_method);
  // CHECK: store { i64, i64 } { i64 ptrtoint (ptr no_cfi @_ZNK1A30unchecked_const_method_int_argEi to i64), i64 0 }, ptr %A5
  auto A5 = (int(CFI_UNCHECKED_CALLEE A::*)(int) const)(&A::unchecked_const_method_int_arg);
}
