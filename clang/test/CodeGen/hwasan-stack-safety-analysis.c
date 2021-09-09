// REQUIRES: aarch64-registered-target

// RUN: %clang -fno-legacy-pass-manager -fsanitize=hwaddress -target aarch64-linux-gnu -S -emit-llvm -mllvm -hwasan-use-stack-safety=true -mllvm -hwasan-generate-tags-with-calls -O2 %s -o - | FileCheck %s --check-prefix=SAFETY
// RUN: %clang -fno-legacy-pass-manager -fsanitize=hwaddress -target aarch64-linux-gnu -S -emit-llvm -mllvm -hwasan-use-stack-safety=false -mllvm -hwasan-generate-tags-with-calls -O2 %s -o - | FileCheck %s --check-prefix=NOSAFETY

// RUN: %clang -flegacy-pass-manager -fsanitize=hwaddress -target aarch64-linux-gnu -S -emit-llvm -mllvm -hwasan-use-stack-safety=true -mllvm -hwasan-generate-tags-with-calls -O2 %s -o - | FileCheck %s --check-prefix=SAFETY
// RUN: %clang -flegacy-pass-manager -fsanitize=hwaddress -target aarch64-linux-gnu -S -emit-llvm -mllvm -hwasan-use-stack-safety=false -mllvm -hwasan-generate-tags-with-calls -O2 %s -o - | FileCheck %s --check-prefix=NOSAFETY

// Default when optimizing, but not with O0.
// RUN: %clang -fsanitize=hwaddress -target aarch64-linux-gnu -S -emit-llvm -mllvm -hwasan-generate-tags-with-calls -O2 %s -o - | FileCheck %s --check-prefix=SAFETY
// RUN: %clang -fsanitize=hwaddress -target aarch64-linux-gnu -S -emit-llvm -mllvm -hwasan-generate-tags-with-calls -O0 %s -o - | FileCheck %s --check-prefix=NOSAFETY

int main(int argc, char **argv) {
  char buf[10];
  volatile char *x = buf;
  *x = 0;
  return buf[0];
  // NOSAFETY: call i8 @__hwasan_generate_tag
  // SAFETY-NOT: call i8 @__hwasan_generate_tag
}
