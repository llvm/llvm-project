// RUN: clang-refactor-test perform -action extract -selected=extract %s | FileCheck %s
;
void extractInline(int x) { // CHECK: "inline int extracted(int x) {\nreturn x + 1;\n}\n\n" [[@LINE]]:1
// extract-begin: +1:11
  int y = x + 1;
// extract-end: -1:16
}
