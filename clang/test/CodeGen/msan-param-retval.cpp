// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=memory -no-enable-noundef-analysis -o - %s | \
// RUN:     FileCheck %s --check-prefixes=CLEAN,CHECK
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=memory -fno-sanitize-memory-param-retval -o - %s | \
// RUN:     FileCheck %s --check-prefixes=NOUNDEF,NOUNDEF_ONLY,CHECK
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=memory -mllvm -msan-eager-checks -o - %s | \
// RUN:     FileCheck %s --check-prefixes=NOUNDEF,EAGER,CHECK
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=memory -no-enable-noundef-analysis -fsanitize-memory-param-retval -o - %s | \
// RUN:     FileCheck %s --check-prefixes=CLEAN,CHECK
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=memory -o - %s | \
// RUN:     FileCheck %s --check-prefixes=NOUNDEF,EAGER,CHECK

void bar(int x) {
}

// CLEAN:   define dso_local void @_Z3bari(i32 %x) #0 {
// NOUNDEF: define dso_local void @_Z3bari(i32 noundef %x) #0 {
// CLEAN:        @__msan_param_tls
// NOUNDEF_ONLY: @__msan_param_tls
// EAGER-NOT:    @__msan_param_tls
// CHECK: }

int foo() {
  return 1;
}

// CLEAN:   define dso_local i32 @_Z3foov() #0 {
// NOUNDEF: define dso_local noundef i32 @_Z3foov() #0 {
// CLEAN:        @__msan_retval_tls
// NOUNDEF_ONLY: @__msan_retval_tls
// EAGER-NOT:    @__msan_retval_tls
// CHECK: }

int noret() {
}

// CLEAN: define dso_local i32 @_Z5noretv() #0 {
// NOUNDEF: define dso_local noundef i32 @_Z5noretv() #0 {
// CHECK:  unreachable
// CHECK: }