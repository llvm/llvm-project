// REQUIRES: aarch64-registered-target

// RUN: %clang -fsanitize-trap=undefined -fsanitize=hwaddress,array-bounds -target aarch64-linux-gnu -S -emit-llvm -mllvm -hwasan-use-stack-safety=true -mllvm -hwasan-generate-tags-with-calls -O2 %s -o - | FileCheck %s --check-prefixes=CHECK,SAFETY
// RUN: %clang -fsanitize-trap=undefined -fsanitize=hwaddress,array-bounds -target aarch64-linux-gnu -S -emit-llvm -mllvm -hwasan-use-stack-safety=false -mllvm -hwasan-generate-tags-with-calls -O2 %s -o - | FileCheck %s --check-prefixes=CHECK,NOSAFETY

// Make sure that HWAsan does not re-check what has been validated by array-bounds

void f(char*);

int foo(unsigned int idx) {
  char buf[10];
  f(buf);
  return buf[idx];
  // CHECK-LABEL: {{.*}}foo
  // NOSAFETY: call void @llvm.hwasan.check.memaccess.shortgranules
  // SAFETY-NOT: call void @llvm.hwasan.check.memaccess.shortgranules
}

int bar(int idx) {
  char buf[10];
  f(buf);
  return buf[idx];
  // CHECK-LABEL: {{.*}}bar
  // NOSAFETY: call void @llvm.hwasan.check.memaccess.shortgranules
  // SAFETY-NOT: call void @llvm.hwasan.check.memaccess.shortgranules
}
