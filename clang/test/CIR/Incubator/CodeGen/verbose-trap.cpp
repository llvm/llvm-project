// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --input-file=%t.ogcg.ll %s --check-prefix=OGCG

// Test basic verbose trap with simple string literals
void test_basic() {
  __builtin_verbose_trap("Security", "Buffer overflow detected");
  // CIR: cir.trap
  // LLVM: call void @llvm.trap()
  // OGCG: call void @llvm.trap()
}

// Test with different category and message
void test_different_messages() {
  __builtin_verbose_trap("Assertion", "x != nullptr");
  // CIR: cir.trap
  // LLVM: call void @llvm.trap()
  // OGCG: call void @llvm.trap()
}

// Test with constexpr string pointers
constexpr const char* kCategory = "Performance";
constexpr const char* kMessage = "Unexpected slow path";

void test_constexpr() {
  __builtin_verbose_trap(kCategory, kMessage);
  // CIR: cir.trap
  // LLVM: call void @llvm.trap()
  // OGCG: call void @llvm.trap()
}

// Test that trap acts as a terminator (code after is unreachable)
void test_terminator() {
  __builtin_verbose_trap("Error", "Invalid state");
  // CIR: cir.trap
  // LLVM: call void @llvm.trap()
  // OGCG: call void @llvm.trap()

  // The following code should still be in the IR but unreachable
  int x = 42; // CIR: cir.store
  (void)x;
}

// Test multiple traps in the same function
void test_multiple_traps(bool condition) {
  if (condition) {
    __builtin_verbose_trap("Branch1", "First trap");
    // CIR: cir.trap
    // LLVM: call void @llvm.trap()
    // OGCG: call void @llvm.trap()
  } else {
    __builtin_verbose_trap("Branch2", "Second trap");
    // CIR: cir.trap
    // LLVM: call void @llvm.trap()
    // OGCG: call void @llvm.trap()
  }
}
