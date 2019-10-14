// REQUIRES: crash-recovery, shell

// FIXME: This XFAIL is cargo-culted from crash-report.c. Do we need it?
// XFAIL: windows-gnu

// RUN: rm -rf %t
// RUN: mkdir -p %t/i %t/m %t

// RUN: env FORCE_CLANG_DIAGNOSTICS_CRASH= TMPDIR=%t TEMP=%t TMP=%t \
// RUN: not %clang -fsyntax-only -nostdinc %s \
// RUN:     -I %S/Inputs/crash-recovery/usr/include -isysroot %/t/i/ \
// RUN:     -fmodules -fmodules-cache-path=%t/m/ 2>&1 | FileCheck %s

// RUN: FileCheck --check-prefix=CHECKSRC %s -input-file %t/crash-vfs-*.m
// RUN: FileCheck --check-prefix=CHECKSH %s -input-file %t/crash-vfs-*.sh
// RUN: FileCheck --check-prefix=CHECKYAML %s -input-file \
// RUN: %t/crash-vfs-*.cache/vfs/vfs.yaml
// RUN: find %t/crash-vfs-*.cache/vfs | \
// RUN:   grep "Inputs/crash-recovery/usr/include/stdio.h" | count 1

#include <stdio.h>

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}.m
// CHECK-NEXT: note: diagnostic msg: {{.*}}.cache

// CHECKSRC: #pragma clang module import cstd.stdio

// CHECKSH: # Crash reproducer
// CHECKSH-NEXT: # Driver args: "-fsyntax-only"
// CHECKSH-NEXT: # Original command: {{.*$}}
// CHECKSH-NEXT: "-cc1"
// CHECKSH: "-resource-dir"
// CHECKSH: "-isysroot" "{{[^"]*}}/i/"
// CHECKSH: "crash-vfs-{{[^ ]*}}.m"
// CHECKSH: "-ivfsoverlay" "crash-vfs-{{[^ ]*}}.cache/vfs/vfs.yaml"
// CHECKSH: "-fmodules-cache-path=crash-vfs-{{[^ ]*}}.cache/repro-modules"

// CHECKYAML: 'case-sensitive':
// CHECKYAML-NEXT: 'use-external-names': 'false',
// CHECKYAML-NEXT: 'overlay-relative': 'true',
// CHECKYAML: 'type': 'directory'
// CHECKYAML: 'name': "/[[PATH:.*]]/Inputs/crash-recovery/usr/include",
// CHECKYAML-NEXT: 'contents': [
// CHECKYAML-NEXT:   {
// CHECKYAML-NEXT:     'type': 'file',
// CHECKYAML-NEXT:     'name': "module.map",
// CHECKYAML-NOT:      'external-contents': "{{[^ ]*}}.cache
// CHECKYAML-NEXT:     'external-contents': "/[[PATH]]/Inputs/crash-recovery/usr/include/module.map"
// CHECKYAML-NEXT:   },

// Test that reading the YAML file will yield the correct path after
// the overlay dir is prefixed to access headers in .cache/vfs directory.

// RUN: unset FORCE_CLANG_DIAGNOSTICS_CRASH
// RUN: %clang -E %s -I %S/Inputs/crash-recovery/usr/include -isysroot %/t/i/ \
// RUN:     -ivfsoverlay %t/crash-vfs-*.cache/vfs/vfs.yaml -fmodules \
// RUN:     -fmodules-cache-path=%t/m/ 2>&1 \
// RUN:     | FileCheck %s --check-prefix=CHECKOVERLAY

// CHECKOVERLAY: #pragma clang module import cstd.stdio /* clang -E: implicit import
