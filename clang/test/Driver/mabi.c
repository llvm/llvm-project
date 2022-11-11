// RUN: %clang --target=i386-unknown-linux -mabi=ms -S %s -### 2>&1 | FileCheck --check-prefix=CHECK %s

int f() {
  // CHECK: warning: argument unused during compilation: '-mabi=ms'
  return 0;
}
