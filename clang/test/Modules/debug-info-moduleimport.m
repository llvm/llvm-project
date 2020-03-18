// RUN: rm -rf %t
// RUN: %clang_cc1 -debug-info-kind=limited -fmodules \
// RUN:     -DGREETING="Hello World" -UNDEBUG \
// RUN:     -fimplicit-module-maps -fmodules-cache-path=%t %s \
// RUN:     -I %S/Inputs -isysroot /tmp/.. -I %t -emit-llvm -o - \
// RUN:     | FileCheck %s --check-prefix=NOIMPORT

// NOIMPORT-NOT: !DIImportedEntity
// NOIMPORT-NOT: !DIModule

// RUN: rm -rf %t
// RUN: %clang_cc1 -debug-info-kind=limited -fmodules \
// RUN:    -DGREETING="Hello World" -UNDEBUG \
// RUN:    -fimplicit-module-maps -fmodules-cache-path=%t %s \
// RUN:    -I %S/Inputs -isysroot /tmp/.. -I %t -emit-llvm \
// RUN:    -debugger-tuning=lldb -o - | FileCheck %s

// CHECK: ![[CU:.*]] = distinct !DICompileUnit
// CHECK-SAME:  sysroot: "/tmp/..")
@import DebugObjC;
// CHECK: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[CU]],
// CHECK-SAME:              entity: ![[MODULE:.*]], file: ![[F:[0-9]+]],
// CHECK-SAME:              line: [[@LINE-3]])
// CHECK: ![[MODULE]] = !DIModule(scope: null, name: "DebugObjC",
// CHECK-SAME:  configMacros: "\22-DGREETING=Hello World\22 \22-UNDEBUG\22",
// CHECK-SAME:  includePath: "{{.*}}test{{.*}}Modules{{.*}}Inputs"
// CHECK: ![[F]] = !DIFile(filename: {{.*}}debug-info-moduleimport.m

// RUN: %clang_cc1 -debug-info-kind=limited -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t %s -I %S/Inputs -isysroot /tmp/.. -I %t \
// RUN:   -emit-llvm -o - | FileCheck %s --check-prefix=NO-SKEL-CHECK
// NO-SKEL-CHECK: distinct !DICompileUnit
// NO-SKEL-CHECK-NOT: distinct !DICompileUnit

// RUN: %clang_cc1 -debug-info-kind=limited -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t -fdebug-prefix-map=%t=/MODULE-CACHE \
// RUN:   -fdebug-prefix-map=%S=/SRCDIR \
// RUN:   -fmodule-format=obj -dwarf-ext-refs \
// RUN:   %s -I %S/Inputs -isysroot /tmp/.. -I %t -emit-llvm -o - \
// RUN:     | FileCheck %s --check-prefix=SKEL-CHECK
// SKEL-CHECK: includePath: "/SRCDIR/Inputs"
// SKEL-CHECK: distinct !DICompileUnit({{.*}}file: ![[CUFILE:[0-9]+]]
// SKEL-CHECK: ![[CUFILE]] = !DIFile({{.*}}directory: "[[COMP_DIR:.*]]"
// SKEL-CHECK: distinct !DICompileUnit({{.*}}file: ![[DWOFILE:[0-9]+]]{{.*}}splitDebugFilename: "/MODULE-CACHE{{.*}}dwoId
// SKEL-CHECK: ![[DWOFILE]] = !DIFile({{.*}}directory: "[[COMP_DIR]]"
