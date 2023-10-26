// REQUIRES: crash-recovery, shell, system-darwin
//
// RUN: rm -rf %t
// RUN: mkdir -p %t/m %t/out

// RUN: %clang_cc1 -x objective-c-header -emit-pch %S/Inputs/pch-used.h \
// RUN:     -o %t/out/pch-used.h.pch -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t/cache -O0 \
// RUN:     -isystem %S/Inputs/System/usr/include

// RUN: env FORCE_CLANG_DIAGNOSTICS_CRASH= TMPDIR=%t TEMP=%t TMP=%t \
// RUN: not %clang %s -E -include-pch %t/out/pch-used.h.pch -fmodules -nostdlibinc \
// RUN:     -fimplicit-module-maps -fmodules-cache-path=%t/cache -O0 \
// RUN:     -Xclang -fno-validate-pch -isystem %S/Inputs/System/usr/include \
// RUN:     -o %t/output.E 2>&1 | FileCheck %s

// RUN: FileCheck --check-prefix=CHECKSH %s -input-file %t/crash-vfs-*.sh
// RUN: FileCheck --check-prefix=CHECKYAML %s -input-file \
// RUN:   %t/crash-vfs-*.cache/vfs/vfs.yaml

void f() { SPXTrace(); }
void g() { double x = DBL_MAX; }

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}.m
// CHECK-NEXT: note: diagnostic msg: {{.*}}.cache

// CHECKSH: "-include-pch" "/[[INCPATH:.*]]/out/pch-used.h.pch"
// CHECKSH: "crash-vfs-{{[^ ]*}}.m"
// CHECKSH: "-ivfsoverlay" "crash-vfs-{{[^ ]*}}.cache/vfs/vfs.yaml"
// CHECKSH: "-fmodules-cache-path=crash-vfs-{{[^ ]*}}.cache/repro-modules"

// CHECKYAML: 'case-sensitive':
// CHECKYAML-NEXT: 'use-external-names': 'false',
// CHECKYAML-NEXT: 'overlay-relative': 'true',
// CHECKYAML: 'type': 'directory'
// CHECKYAML: 'name': "/[[PATH:.*]]/out",
// CHECKYAML-NEXT: 'contents': [
// CHECKYAML-NEXT:   {
// CHECKYAML-NEXT:     'type': 'file',
// CHECKYAML-NEXT:     'name': "pch-used.h.pch",
// CHECKYAML-NEXT:     'external-contents': "/[[PATH]]/out/pch-used.h.pch"
