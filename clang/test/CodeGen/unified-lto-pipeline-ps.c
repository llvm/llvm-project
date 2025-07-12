// RUN: %clang -flto=thin --target=x86_64-scei-ps4 -O2 -c %s -o %t.ps4.tu -Xclang -fdebug-pass-manager 2>%t.ps4.tu.txt
// RUN: %clang -flto=thin --target=x86_64-sie-ps5  -O2 -c %s -o %t.ps5.tu -Xclang -fdebug-pass-manager 2>%t.ps5.tu.txt
// RUN: %clang -flto=full --target=x86_64-scei-ps4 -O2 -c %s -o %t.ps4.fu -Xclang -fdebug-pass-manager 2>%t.ps4.fu.txt
// RUN: %clang -flto=full --target=x86_64-sie-ps5  -O2 -c %s -o %t.ps5.fu -Xclang -fdebug-pass-manager 2>%t.ps5.fu.txt
// RUN: %clang -flto=thin -fno-unified-lto --target=x86_64-scei-ps4 -O2 -c %s -o %t.ps4.tn -Xclang -fdebug-pass-manager 2>%t.ps4.tn.txt
// RUN: %clang -flto=full -fno-unified-lto --target=x86_64-scei-ps4 -O2 -c %s -o %t.ps4.fn -Xclang -fdebug-pass-manager 2>%t.ps4.fn.txt
// RUN: %clang -flto=thin --target=x86_64-unknown-linux -O2 -c %s -o %t.l.tn -Xclang -fdebug-pass-manager 2>%t.l.tn.txt
// RUN: %clang -flto=full --target=x86_64-unknown-linux -O2 -c %s -o %t.l.fn -Xclang -fdebug-pass-manager 2>%t.l.fn.txt
/// Pre-link bitcode and pass pipelines should be identical for (unified) thin/full on PS4/PS5.
/// Pipeline on PS5 is also identical (but bitcode won't be identical to PS4 due to the embedded triple).
// RUN: cmp %t.ps4.tu %t.ps4.fu
// RUN: cmp %t.ps5.tu %t.ps5.fu
// RUN: diff %t.ps4.tu.txt %t.ps4.fu.txt
// RUN: diff %t.ps4.tu.txt %t.ps5.tu.txt
// RUN: diff %t.ps5.tu.txt %t.ps5.fu.txt
// RUN: FileCheck --input-file %t.ps4.tu.txt %s
// CHECK: ThinLTOBitcodeWriterPass
/// Non-unified PS4/PS5 pass pipelines match Linux. Thin/full are different.
// RUN: not diff %t.ps4.tn.txt %t.ps4.fn.txt
// RUN: diff %t.ps4.tn.txt %t.l.tn.txt
// RUN: diff %t.ps4.fn.txt %t.l.fn.txt
/// PS4/PS5 unified use the full Linux pipeline (except ThinLTOBitcodeWriterPass vs BitcodeWriterPass).
// RUN: not diff -u %t.ps4.tu.txt %t.l.fn.txt | FileCheck %s --check-prefix=DIFF --implicit-check-not="{{^[-+!<>] }}"
// DIFF:      -Running pass: ThinLTOBitcodeWriterPass
// DIFF-NEXT: +Running pass: BitcodeWriterPass

int foo() {
  return 2 + 2;
}

int bar() {
  return foo() + 1;
}
