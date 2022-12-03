// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -fsanitize=memory -no-enable-noundef-analysis -o - %s | \
// RUN:     FileCheck %s --check-prefixes=CLEAN,CHECK
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -fsanitize=memory -fno-sanitize-memory-param-retval -o - %s | \
// RUN:     FileCheck %s --check-prefixes=NOUNDEF,NOUNDEF_ONLY,CHECK
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -fsanitize=memory -mllvm -msan-eager-checks -o - %s | \
// RUN:     FileCheck %s --check-prefixes=NOUNDEF,EAGER,CHECK
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -fsanitize=memory -no-enable-noundef-analysis -fsanitize-memory-param-retval -o - %s | \
// RUN:     FileCheck %s --check-prefixes=CLEAN,CHECK
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -fsanitize=memory -o - %s | \
// RUN:     FileCheck %s --check-prefixes=NOUNDEF,EAGER,CHECK

void bar(int x) {
}

// CLEAN:   define dso_local void @bar(i32 %x) #0 {
// NOUNDEF: define dso_local void @bar(i32 noundef %x) #0 {
// CLEAN:        @__msan_param_tls
// NOUNDEF_ONLY: @__msan_param_tls
// EAGER-NOT:    @__msan_param_tls
// CHECK: }

int foo() {
  return 1;
}

// CHECK: define dso_local i32 @foo() #0 {
// CHECK:   @__msan_retval_tls
// CHECK: }

int noret() {
}

// CHECK: define dso_local i32 @noret() #0 {
// CHECK:   %retval = alloca
// CHECK: }