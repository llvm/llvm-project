// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ogcg.ll %s

// Test addressof builtins in emitPointerWithAlignment context
// This tests the fix for crash at CIRGenExpr.cpp:240 (248 production crashes)

struct S {
  void operator&() = delete;  // Ensures addressof is needed
};

// Test 1: __builtin_addressof in delete expression (main crash scenario)
// CIR-LABEL: @_Z{{.*}}test_delete_builtin
// LLVM-LABEL: @_Z{{.*}}test_delete_builtin
// OGCG-LABEL: @_Z{{.*}}test_delete_builtin
void test_delete_builtin() {
  S* s = new S();
  delete __builtin_addressof(*s);
  // CIR: cir.call @_ZdlPvm
  // LLVM: call{{.*}} @_ZdlPv
  // OGCG: call{{.*}} @_ZdlPv
}

// Test 2: Simple case - local variable
// CIR-LABEL: @_Z{{.*}}test_simple_local
// LLVM-LABEL: @_Z{{.*}}test_simple_local
// OGCG-LABEL: @_Z{{.*}}test_simple_local
int* test_simple_local() {
  int x = 42;
  return __builtin_addressof(x);
  // CIR: cir.alloca
  // CIR: cir.return
  // LLVM: alloca
  // LLVM: ret ptr
  // OGCG: alloca
  // OGCG: ret ptr
}

// Test 3: Global variable
extern int global_var;
// CIR-LABEL: @_Z{{.*}}test_global
// LLVM-LABEL: @_Z{{.*}}test_global
// OGCG-LABEL: @_Z{{.*}}test_global
int* test_global() {
  return __builtin_addressof(global_var);
  // CIR: cir.get_global
  // CIR: cir.return
  // LLVM: @global_var
  // OGCG: @global_var
}

// Test 4: Conditional operator with addressof
// CIR-LABEL: @_Z{{.*}}test_conditional
// LLVM-LABEL: @_Z{{.*}}test_conditional
// OGCG-LABEL: @_Z{{.*}}test_conditional
S *test_conditional(bool b, S &s, S &t) {
  return __builtin_addressof(b ? s : t);
  // CIR: cir.ternary
  // LLVM: phi ptr
  // OGCG: phi ptr
}

// Test 5: Member access
struct Container {
  int value;
};

// CIR-LABEL: @_Z{{.*}}test_member
// LLVM-LABEL: @_Z{{.*}}test_member
// OGCG-LABEL: @_Z{{.*}}test_member
int* test_member(Container& c) {
  return __builtin_addressof(c.value);
  // CIR: cir.get_member
  // CIR: cir.return
  // LLVM: getelementptr
  // LLVM: ret ptr
  // OGCG: getelementptr
  // OGCG: ret ptr
}

// Test 6: Array element
// CIR-LABEL: @_Z{{.*}}test_array_elem
// LLVM-LABEL: @_Z{{.*}}test_array_elem
// OGCG-LABEL: @_Z{{.*}}test_array_elem
int* test_array_elem(int i, int* arr) {
  return __builtin_addressof(arr[i]);
  // CIR: cir.ptr_stride
  // CIR: cir.return
  // LLVM: getelementptr
  // LLVM: ret ptr
  // OGCG: getelementptr
  // OGCG: ret ptr
}

