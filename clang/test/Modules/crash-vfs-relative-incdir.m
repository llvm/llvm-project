// REQUIRES: crash-recovery, shell, system-darwin

// RUN: rm -rf %t
// RUN: mkdir -p %t/m
// RUN: cd %S/Inputs/crash-recovery/usr

// RUN: not env FORCE_CLANG_DIAGNOSTICS_CRASH= TMPDIR=%t TEMP=%t TMP=%t \
// RUN: %clang -fsyntax-only -nostdinc %s -Iinclude \
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
// CHECKSH: "-I" "/[[INCPATH:.*]]/include"
// CHECKSH: "crash-vfs-{{[^ ]*}}.m"
// CHECKSH: "-ivfsoverlay" "crash-vfs-{{[^ ]*}}.cache/vfs/vfs.yaml"
// CHECKSH: "-fmodules-cache-path=crash-vfs-{{[^ ]*}}.cache/repro-modules"

// CHECKYAML: 'case-sensitive':
// CHECKYAML-NEXT: 'use-external-names': 'false',
// CHECKYAML-NEXT: 'overlay-relative': 'true',
// CHECKYAML-NEXT: 'ignore-non-existent-contents': 'false'
// CHECKYAML: 'type': 'directory'
// CHECKYAML: 'name': "/[[PATH:.*]]/Inputs/crash-recovery/usr/include",
// CHECKYAML-NEXT: 'contents': [
// CHECKYAML-NEXT:   {
// CHECKYAML-NEXT:     'type': 'file',
// CHECKYAML-NEXT:     'name': "module.map",
// CHECKYAML-NOT:      'external-contents': "{{[^ ]*}}.cache
// CHECKYAML-NEXT:     'external-contents': "/[[PATH]]/Inputs/crash-recovery/usr/include/module.map"
// CHECKYAML-NEXT:   },

// Run the reproducer script - regular exit code is enough to test it works.
// Note that we don't yet support reusing the modules pcm; what we do
// support is re-building the modules relying solely on the header files dumped
// inside .cache/vfs, mapped by .cache/vfs/vfs.yaml.

// RUN: cd %t
// RUN: chmod 755 crash-vfs-*.sh
// RUN: ./crash-vfs-*.sh
