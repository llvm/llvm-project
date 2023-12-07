// RUN: %clang --target=x86_64-unknown-unknown-linux -Xclang -fdebug-pass-manager -flto=thin -funified-lto -O2 -c %s -o %t.0 2>&1 | FileCheck --check-prefix=THIN %s
// RUN: mv %t.0 %t.1
// RUN: %clang --target=x86_64-unknown-unknown-linux -Xclang -fdebug-pass-manager -flto=full -funified-lto -O2 -c %s -o %t.0 2>&1 | FileCheck --check-prefix=THIN %s
// RUN: %clang --target=x86_64-unknown-unknown-linux -Xclang -fdebug-pass-manager -flto=thin -O2 -c %s -o %t.2 2>&1 | FileCheck --check-prefix=THIN %s
// RUN: mv %t.2 %t.3
// RUN: %clang --target=x86_64-unknown-unknown-linux -Xclang -fdebug-pass-manager -flto=full -O2 -c %s -o %t.2 2>&1 | FileCheck --check-prefix=FULL %s
// RUN: cmp %t.0 %t.1
// THIN: ThinLTOBitcodeWriterPass
// FULL-NOT: ThinLTOBitcodeWriterPass

int foo() {
  return 2 + 2;
}

int bar() {
  return foo() + 1;
}
