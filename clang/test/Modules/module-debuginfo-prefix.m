// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// REQUIRES: asserts

// Modules:
// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -fmodules -fmodule-format=obj \
// RUN:   -fdebug-prefix-map=%S/Inputs=/OVERRIDE \
// RUN:   -fimplicit-module-maps -DMODULES -fmodules-cache-path=%t %s \
// RUN:   -I %S/Inputs -I %t -emit-llvm -o %t.ll \
// RUN:   -mllvm -debug-only=pchcontainer &>%t-mod.ll
// RUN: cat %t-mod.ll | FileCheck %s

// PCH:
// RUN: %clang_cc1 -x objective-c -emit-pch -fmodule-format=obj -I %S/Inputs \
// RUN:   -fdebug-prefix-map=%S/Inputs=/OVERRIDE \
// RUN:   -o %t.pch %S/Inputs/DebugObjC.h \
// RUN:   -mllvm -debug-only=pchcontainer &>%t-pch.ll
// RUN: cat %t-pch.ll | FileCheck %s

#ifdef MODULES
@import DebugObjC;
#endif

// Dir should always be empty, but on Windows we can't recognize /var
// as being an absolute path.
// CHECK: !DIFile(filename: "/OVERRIDE{{/|\\\\?}}DebugObjC.h", directory: "{{()|(.*:.*)}}")
