// RUN: rm -rf %t && mkdir %t && %clang -c -fdiagnostics-add-output=sarif:file=%t%{fs-sep}sarif-log-path.cpp.sarif -Xclang -verify %s
// RUN: cat %t%{fs-sep}sarif-log-path.cpp.sarif | %normalize_sarif | diff -U1 -b %S/Inputs/expected-sarif/sarif-log.cpp.sarif -
// RUN: %clang -c -fdiagnostics-add-output=sarif:version=2.1,file=%t%{fs-sep}sarif-log-path.cpp.1.sarif -fdiagnostics-add-output=sarif:file=%t%{fs-sep}sarif-log-path.cpp.2.sarif -Xclang -verify %s
// RUN: cat %t%{fs-sep}sarif-log-path.cpp.1.sarif | %normalize_sarif | diff -U1 -b %S/Inputs/expected-sarif/sarif-log.cpp.sarif -
// RUN: cat %t%{fs-sep}sarif-log-path.cpp.2.sarif | %normalize_sarif | diff -U1 -b %S/Inputs/expected-sarif/sarif-log.cpp.sarif -

[[deprecated]]  // expected-note{{'depfunc' has been explicitly marked deprecated here}}
void depfunc();

void call_depfunc() {
  depfunc();  // expected-warning{{'depfunc' is deprecated}}
}
