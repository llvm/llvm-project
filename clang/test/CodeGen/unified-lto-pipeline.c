// RUN: %clang --target=x86_64-unknown-unknown-linux -Xclang -fdebug-pass-manager -flto=thin -funified-lto -O2 -c %s -o %t.0 2>%t.0.txt
// RUN: %clang --target=x86_64-unknown-unknown-linux -Xclang -fdebug-pass-manager -flto=full -funified-lto -O2 -c %s -o %t.1 2>%t.1.txt
// RUN: %clang --target=x86_64-unknown-unknown-linux -Xclang -fdebug-pass-manager -flto=thin -O2 -c %s -o %t.2 2>%t.2.txt
// RUN: %clang --target=x86_64-unknown-unknown-linux -Xclang -fdebug-pass-manager -flto=full -O2 -c %s -o %t.3 2>%t.3.txt
// RUN: FileCheck --input-file %t.0.txt %s --check-prefix=THIN
// RUN: FileCheck --input-file %t.3.txt %s --check-prefix=FULL
// THIN: ThinLTOBitcodeWriterPass
// FULL-NOT: ThinLTOBitcodeWriterPass
/// Check that thin/full unified bitcode matches.
// RUN: cmp %t.0 %t.1
/// Check that pass pipelines for thin, thin-unified, full-unified all match.
// RUN: diff %t.0.txt %t.1.txt
// RUN: diff %t.0.txt %t.2.txt
/// Pass pipeline for full is different.
// RUN: not diff %t.0.txt %t.3.txt

int foo() {
  return 2 + 2;
}

int bar() {
  return foo() + 1;
}
