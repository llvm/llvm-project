// RUN: rm -rf %t
// RUN: env CINDEXTEST_COMPLETION_CACHING=1 \
// RUN:     c-index-test -test-load-source-reparse 2 local %s -fmodules -fmodules-cache-path=%t -I %S/Inputs \
// RUN:   | FileCheck %s

// CHECK: complete-module-undef.m:7:1: ModuleImport=ModuleUndef:7:1 (Definition) Extent=[7:1 - 7:20]
@import ModuleUndef;
