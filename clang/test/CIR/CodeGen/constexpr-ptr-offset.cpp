// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct View {
  const char *ptr;
  int len;
  constexpr View(const char *p, int n) : ptr(p), len(n) {}
};

constexpr const char *global_str = "hello";

void test() {
  constexpr View v(global_str + 2, 3);
  (void)v;
}

// CIR-LABEL: @_Z4testv
// CIR: #cir.global_view<@{{.*}}str{{.*}}, [2 : i32]> : !cir.ptr<!s8i>

// LLVM: getelementptr{{.*}}(i8, ptr @{{.*}}str{{.*}}, i64 2)
