// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// Test that an @import inside a module is not represented in the debug info.

// REQUIRES: asserts

// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -fmodules -fmodule-format=obj \
// RUN:   -fimplicit-module-maps -fmodules-cache-path=%t %s \
// RUN:   -debugger-tuning=lldb -I %S/Inputs -emit-llvm -o %t.ll \
// RUN:   -mllvm -debug-only=pchcontainer &>%t-mod.ll
// RUN: cat %t-mod.ll | FileCheck %s

@import DebugObjCImport.SubModule;

// CHECK: distinct !DICompileUnit(language: DW_LANG_ObjC
// CHECK: DW_TAG_structure_type, name: "DebugObjCImport"
// CHECK: ![[HEADER:.*]] = !DIFile(filename: {{.*}}DebugObjCImport.h"
// CHECK: ![[SUBMOD:.*]] = !DIModule({{.*}}name: "SubModule"
// CHECK: !DIImportedEntity(tag: DW_TAG_imported_declaration,
// CHECK-SAME:              scope: ![[SUBMOD]], entity: ![[EMPTY:[0-9]+]],
// CHECK-SAME:              file: ![[HEADER]], line: 1)
// CHECK: ![[EMPTY]] = !DIModule(scope: null, name: "Empty"
