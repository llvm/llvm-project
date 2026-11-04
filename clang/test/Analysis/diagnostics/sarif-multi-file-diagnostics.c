// RUN: rm -rf %t && mkdir %t
// RUN: %clang_analyze_cc1 -analyzer-checker=core %s -verify -analyzer-output=sarif-html -o %t%{fs-sep}out1.sarif
// RUN: %clang_analyze_cc1 -analyzer-checker=core %s -verify -analyzer-output=sarif-html -o %t%{fs-sep}out2.sarif
// RUN: cat %t%{fs-sep}out1.sarif | %normalize_sarif | diff -U1 -b %S/Inputs/expected-sarif/sarif-multi-file-diagnostics.c.sarif -
// RUN: cat %t%{fs-sep}out2.sarif | %normalize_sarif | diff -U1 -b %S/Inputs/expected-sarif/sarif-multi-file-diagnostics.c.sarif -

int test(int *p) {
  if (p)
    return 0;
  else
    return *p;  // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
}
