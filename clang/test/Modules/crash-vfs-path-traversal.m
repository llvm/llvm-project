// REQUIRES: crash-recovery, shell, non-ms-sdk, non-ps4-sdk

// FIXME: Canonicalizing paths to remove relative traversal components
// currenty fails a unittest on windows and is disable by default.
// FIXME: This XFAIL is cargo-culted from crash-report.c. Do we need it?
// XFAIL: windows-gnu

// RUN: rm -rf %t
// RUN: mkdir -p %t/i %t/m %t

// RUN: not env FORCE_CLANG_DIAGNOSTICS_CRASH= TMPDIR=%t TEMP=%t TMP=%t \
// RUN: %clang -fsyntax-only %s -I %S/Inputs/crash-recovery -isysroot %/t/i/    \
// RUN: -fmodules -fmodules-cache-path=%t/m/ 2>&1 | FileCheck %s

// RUN: FileCheck --check-prefix=CHECKSRC %s -input-file %t/crash-vfs-*.m
// RUN: FileCheck --check-prefix=CHECKSH %s -input-file %t/crash-vfs-*.sh
// RUN: FileCheck --check-prefix=CHECKYAML %s -input-file \
// RUN: %t/crash-vfs-*.cache/vfs/vfs.yaml
// RUN: find %t/crash-vfs-*.cache/vfs | \
// RUN:   grep "Inputs/crash-recovery/usr/include/stdio.h" | count 1

#include "usr/././//////include/../include/./././../include/stdio.h"

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}.m
// CHECK-NEXT: note: diagnostic msg: {{.*}}.cache

// CHECKSRC: #pragma clang module import cstd.stdio

// CHECKSH: # Crash reproducer
// CHECKSH-NEXT: # Driver args: "-fsyntax-only"
// CHECKSH-NEXT: # Original command: {{.*$}}
// CHECKSH-NEXT: "-cc1"
// CHECKSH: "-isysroot" "{{[^"]*}}/i/"
// CHECKSH-NOT: "-fmodules-cache-path="
// CHECKSH: "crash-vfs-{{[^ ]*}}.m"
// CHECKSH: "-ivfsoverlay" "crash-vfs-{{[^ ]*}}.cache/vfs/vfs.yaml"
// CHECKSH: "-fmodules-cache-path=crash-vfs-{{[^ ]*}}.cache/repro-modules"

// CHECKYAML: 'case-sensitive':
// CHECKYAML-NEXT: 'use-external-names': 'false',
// CHECKYAML-NEXT: 'overlay-relative': 'true',
// CHECKYAML:     'type': 'directory'
// CHECKYAML:     'name': "/[[PATH:.*]]/Inputs/crash-recovery/usr/include",
// CHECKYAML-NEXT: 'contents': [
// CHECKYAML-NEXT:   {
// CHECKYAML-NEXT:     'type': 'file',
// CHECKYAML-NEXT:     'name': "module.map",
// CHECKYAML-NEXT:     'external-contents': "/[[PATH]]/Inputs/crash-recovery/usr/include/module.map"
// CHECKYAML-NEXT:   },

// Replace the paths in the YAML files with relative ".." traversals
// and fed into clang to test whether we're correctly representing them
// in the VFS overlay.

// RUN: sed -e "s@usr/include@usr/include/../include@g" \
// RUN:     %t/crash-vfs-*.cache/vfs/vfs.yaml > %t/vfs.yaml
// RUN: cp %t/vfs.yaml %t/crash-vfs-*.cache/vfs/vfs.yaml
// RUN: unset FORCE_CLANG_DIAGNOSTICS_CRASH
// RUN: %clang -E %s -I %S/Inputs/crash-recovery -isysroot %/t/i/ \
// RUN:     -ivfsoverlay %t/crash-vfs-*.cache/vfs/vfs.yaml -fmodules \
// RUN:     -fmodules-cache-path=%t/m/ 2>&1 \
// RUN:     | FileCheck %s --check-prefix=CHECKOVERLAY

// CHECKOVERLAY: #pragma clang module import cstd.stdio /* clang -E: implicit import
