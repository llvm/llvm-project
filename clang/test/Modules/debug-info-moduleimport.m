// RUN: rm -rf %t
// RUN: %clang_cc1 -debug-info-kind=limited -fmodules -DGREETING="Hello World" -UNDEBUG -fimplicit-module-maps -fmodules-cache-path=%t %s -I %S/Inputs -isysroot /tmp/.. -I %t -emit-llvm -o - | FileCheck %s --check-prefix=NOIMPORT

// NOIMPORT-NOT: !DIImportedEntity
// NOIMPORT-NOT: !DIModule

// RUN: rm -rf %t
// RUN: %clang_cc1 -debug-info-kind=limited -fmodules -DGREETING="Hello World" -UNDEBUG -fimplicit-module-maps -fmodules-cache-path=%t %s -I %S/Inputs -isysroot /tmp/.. -I %t -emit-llvm -debugger-tuning=lldb -o - | FileCheck %s

// CHECK: ![[CU:.*]] = distinct !DICompileUnit
@import DebugObjC;
// CHECK: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[CU]],
// CHECK-SAME:              entity: ![[MODULE:.*]], line: [[@LINE-2]])
// CHECK: ![[MODULE]] = !DIModule(scope: null, name: "DebugObjC",
// CHECK-SAME:  configMacros: "\22-DGREETING=Hello World\22 \22-UNDEBUG\22",
// CHECK-SAME:  includePath: "{{.*}}test{{.*}}Modules{{.*}}Inputs",
// CHECK-SAME:  isysroot: "/tmp/..")


// RUN: %clang_cc1 -debug-info-kind=limited -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:   %s -I %S/Inputs -isysroot /tmp/.. -I %t -emit-llvm -o - \
// RUN:     | FileCheck %s --check-prefix=NO-SKEL-CHECK
// NO-SKEL-CHECK: distinct !DICompileUnit
// NO-SKEL-CHECK-NOT: distinct !DICompileUnit

// RUN: %clang_cc1 -debug-info-kind=limited -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:   -fmodule-format=obj -dwarf-ext-refs \
// RUN:   %s -I %S/Inputs -isysroot /tmp/.. -I %t -emit-llvm -o - \
// RUN:     | FileCheck %s --check-prefix=SKEL-CHECK
// SKEL-CHECK: distinct !DICompileUnit
// SKEL-CHECK: distinct !DICompileUnit{{.*}}dwoId
