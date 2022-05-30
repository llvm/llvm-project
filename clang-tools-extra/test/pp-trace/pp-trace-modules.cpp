// RUN: rm -rf %t
// RUN: pp-trace -callbacks '*,-FileChanged,-MacroDefined' %s -- -x objective-c++ -undef -target x86_64 -std=c++11 -fmodules -fmodules-cache-path=%t -I%S -I%S/Input | FileCheck --strict-whitespace %s

// CHECK: ---

@import Level1A;

// CHECK-NEXT: - Callback: moduleImport
// CHECK-NEXT:   ImportLoc: "{{.*}}{{[/\\]}}pp-trace-modules.cpp:[[@LINE-3]]:2"
// CHECK-NEXT:   Path: [{Name: Level1A, Loc: "{{.*}}{{[/\\]}}pp-trace-modules.cpp:[[@LINE-4]]:9"}]
// CHECK-NEXT:   Imported: Level1A

@import Level1B.Level2B;

// CHECK-NEXT: - Callback: moduleImport
// CHECK-NEXT:   ImportLoc: "{{.*}}{{[/\\]}}pp-trace-modules.cpp:[[@LINE-3]]:2"
// CHECK-NEXT:   Path: [{Name: Level1B, Loc: "{{.*}}{{[/\\]}}pp-trace-modules.cpp:[[@LINE-4]]:9"}, {Name: Level2B, Loc: "{{.*}}{{[/\\]}}pp-trace-modules.cpp:[[@LINE-4]]:17"}]
// CHECK-NEXT:   Imported: Level2B
// CHECK-NEXT: - Callback: EndOfMainFile
// CHECK-NEXT: ...
