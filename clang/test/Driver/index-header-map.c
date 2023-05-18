// RUN: %clang -I%S/Before -index-header-map -I%S/Index -I%S/After %s -### 2>> %t.log
// RUN: FileCheck %s < %t.log

// CHECK: {{-I.*Before.*-index-header-map.*-I.*Index.*-I.*After}}
