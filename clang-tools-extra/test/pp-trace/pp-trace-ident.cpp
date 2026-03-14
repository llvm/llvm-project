// RUN: pp-trace -callbacks '*,-FileChanged,-MacroDefined' %s -- -undef -target x86_64 -std=c++11 | FileCheck --strict-whitespace %s

#ident "$Id$"

// CHECK: ---
// CHECK-NEXT: - Callback: Ident
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-ident.cpp:3:2"
// CHECK-NEXT:   Str: "$Id$"
// CHECK-NEXT: - Callback: EndOfMainFile
// CHECK-NEXT: ...
