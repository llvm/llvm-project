// RUN: rm -rf %t
// RUN: env CINDEXTEST_COMPLETION_CACHING=1 \
// RUN:     c-index-test -test-load-source-reparse 2 local %s -fmodules -fmodules-cache-path=%t -I %S/Inputs \
// RUN:   | FileCheck %s

// rdar://18416901 (used to crash)
// CHECK: complete-module-undef.m:8:1: ModuleImport=ModuleUndef:8:1 (Definition) Extent=[8:1 - 8:20]
@import ModuleUndef;
