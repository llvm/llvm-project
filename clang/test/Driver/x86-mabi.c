// RUN: %clang -### --target=x86_64-windows-msvc -mabi=ms -S %s 2>&1 | FileCheck %s
// RUN: not %clang -### --target=i386-unknown-linux -mabi=ms -S %s 2>&1 | FileCheck --check-prefix=ERR %s
// RUN: not %clang -### --target=x86_64-windows-msvc -mabi=sysv -S %s 2>&1 | FileCheck --check-prefix=ERR %s
// RUN: %clang -### --target=i386-unknown-linux -mabi=sysv -S %s 2>&1 | FileCheck %s

// RUN: %clang -### --target=x86_64-windows-gnu -mabi=ms -S %s 2>&1 | FileCheck %s

// CHECK-NOT: {{error|warning}}:
// ERR: error: unsupported option '-mabi=' for target '{{.*}}'

int f() {
  return 0;
}
