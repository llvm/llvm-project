// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.og.ll %s

// Test basic thread_local variable with constant initialization
thread_local int tls_const = 42;
// CIR: cir.global{{.*}}tls_dyn{{.*}}@tls_const = #cir.int<42> : !s32i
// LLVM: @tls_const = thread_local global i32 42
// OGCG: @tls_const = thread_local global i32 42

// Test __thread (GNU-style) thread_local
__thread int tls_gnu_style = 10;
// CIR: cir.global{{.*}}tls_dyn{{.*}}@tls_gnu_style = #cir.int<10> : !s32i
// LLVM: @tls_gnu_style = thread_local global i32 10
// OGCG: @tls_gnu_style = thread_local global i32 10

// Test thread_local function-local static (constant init)
int get_tls_static() {
  thread_local int tls_func_static = 100;
  return ++tls_func_static;
}
// CIR-LABEL: cir.func{{.*}}@_Z14get_tls_staticv
// CIR: cir.get_global{{.*}}@_ZZ14get_tls_staticvE15tls_func_static
// LLVM-LABEL: @_Z14get_tls_staticv
// LLVM: load{{.*}}@_ZZ14get_tls_staticvE15tls_func_static
// OGCG-LABEL: @_Z14get_tls_staticv
// OGCG: @llvm.threadlocal.address.p0(ptr{{.*}}@_ZZ14get_tls_staticvE15tls_func_static)

// Test reading from thread_local variable
int read_tls() {
  return tls_const;
}
// CIR-LABEL: cir.func{{.*}}@_Z8read_tlsv
// CIR: cir.get_global thread_local @tls_const
// LLVM-LABEL: @_Z8read_tlsv
// LLVM: @llvm.threadlocal.address.p0(ptr @tls_const)
// OGCG-LABEL: @_Z8read_tlsv
// OGCG: @llvm.threadlocal.address.p0(ptr{{.*}}@tls_const)

// Test writing to thread_local variable
void write_tls(int val) {
  tls_const = val;
}
// CIR-LABEL: cir.func{{.*}}@_Z9write_tlsi
// CIR: cir.get_global thread_local @tls_const
// CIR: cir.store
// LLVM-LABEL: @_Z9write_tlsi
// LLVM: @llvm.threadlocal.address.p0(ptr @tls_const)
// OGCG-LABEL: @_Z9write_tlsi
// OGCG: @llvm.threadlocal.address.p0(ptr{{.*}}@tls_const)

// Test extern thread_local
extern thread_local int tls_extern;
int use_extern_tls() {
  return tls_extern;
}
// CIR-LABEL: cir.func{{.*}}@_Z14use_extern_tlsv
// CIR: cir.get_global thread_local @tls_extern
// LLVM-LABEL: @_Z14use_extern_tlsv
// LLVM: @llvm.threadlocal.address.p0(ptr @tls_extern)
// OGCG-LABEL: @_Z14use_extern_tlsv
// OGCG: call ptr @_ZTW10tls_extern()
