// RUN: rm -rf %t.h.gch
// RUN: mkdir -p %t.h.gch
//
// RUN: %clang -x c-header %S/pch-dir.h -DFOO=foo -o %t.h.gch/c.gch 
// RUN: %clang -x c-header %S/pch-dir.h -DFOO=bar -o %t.h.gch/cbar.gch 
// RUN: %clang -x c++-header -std=c++98 %S/pch-dir.h -o %t.h.gch/cpp.gch 
// RUN: %clang -include %t.h -DFOO=foo -fsyntax-only %s -Xclang -print-stats 2> %t.clog
// RUN: FileCheck -check-prefix=CHECK-C %s < %t.clog
// RUN: %clang -include %t.h -DFOO=bar -DBAR=bar -fsyntax-only %s -Xclang -ast-print > %t.cbarlog
// RUN: FileCheck -check-prefix=CHECK-CBAR %s < %t.cbarlog
// RUN: %clang -x c++ -include %t.h -std=c++98 -fsyntax-only %s -Xclang -print-stats 2> %t.cpplog
// RUN: FileCheck -check-prefix=CHECK-CPP %s < %t.cpplog

// RUN: not %clang -x c++ -std=c++11 -include %t.h -fsyntax-only %s 2> %t.cpp11log
// RUN: FileCheck -check-prefix=CHECK-NO-SUITABLE %s < %t.cpp11log

// Don't crash if the precompiled header file is missing.
// RUN: not %clang_cc1 -include-pch %t.h.gch -DFOO=baz -fsyntax-only %s -print-stats 2> %t.missinglog
// RUN: FileCheck -check-prefix=CHECK-NO-SUITABLE %s < %t.missinglog

// CHECK-CBAR: int bar
int FOO;

int get(void) {
#ifdef __cplusplus
  // CHECK-CPP: .h.gch{{[/\\]}}cpp.gch
  return i;
#else
  // CHECK-C: .h.gch{{[/\\]}}c.gch
  return j;
#endif
}

// CHECK-NO-SUITABLE: no suitable precompiled header file found in directory
