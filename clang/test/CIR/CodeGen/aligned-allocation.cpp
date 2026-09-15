// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct alignas(32) Big {
  char data[80];
};

void f() {
  Big *p = new Big;
  delete p;
}

// CIR-LABEL: cir.func{{.*}}@_Z1fv
// CIR:         %[[SIZE_NEW:[0-9]+]] = cir.const #cir.int<96> : !u64i
// CIR:         %[[ALIGN_NEW:[0-9]+]] = cir.const #cir.int<32> : !u64i
// CIR:         cir.call @_ZnwmSt11align_val_t(%[[SIZE_NEW]], %[[ALIGN_NEW]])
// CIR:         %[[SIZE_DEL:[0-9]+]] = cir.const #cir.int<96> : !u64i
// CIR:         %[[ALIGN_DEL:[0-9]+]] = cir.const #cir.int<32> : !u64i
// CIR:         cir.call @_ZdlPvmSt11align_val_t(%{{.+}}, %[[SIZE_DEL]], %[[ALIGN_DEL]])

// LLVM-LABEL: define{{.*}}void @_Z1fv
// LLVM:         call {{.*}}ptr @_ZnwmSt11align_val_t(i64 {{[^,]*}}96, i64 {{[^,]*}}32)
// LLVM:         call void @_ZdlPvmSt11align_val_t(ptr {{[^,]*%[^,]+}}, i64 {{[^,]*}}96, i64 {{[^,]*}}32)
