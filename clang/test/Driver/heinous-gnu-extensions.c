// RUN: %clang -### -fsyntax-only -fheinous-gnu-extensions %s 2>&1 | FileCheck %s

// CHECK: -Wno-error=invalid-gnu-asm-cast

int main(void) {}
