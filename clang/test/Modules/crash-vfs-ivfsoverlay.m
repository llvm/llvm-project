// REQUIRES: crash-recovery, shell, system-darwin

// RUN: rm -rf %t
// RUN: mkdir -p %t/m
// RUN: cp %S/../VFS/Inputs/actual_module2.map %t/actual_module2.map
// RUN: sed -e "s:INPUT_DIR:%t:g" -e "s:OUT_DIR:%t/example:g" \
// RUN:   %S/../VFS/Inputs/vfsoverlay2.yaml > %t/srcvfs.yaml

// RUN: not env FORCE_CLANG_DIAGNOSTICS_CRASH= TMPDIR=%t TEMP=%t TMP=%t \
// RUN: %clang -fsyntax-only -nostdinc %s \
// RUN:     -I %S/Inputs/crash-recovery/usr/include \
// RUN:     -ivfsoverlay %t/srcvfs.yaml \
// RUN:     -fmodules -fmodules-cache-path=%t/m/ 2>&1 | FileCheck %s

// RUN: FileCheck --check-prefix=CHECKSH %s -input-file %t/crash-vfs-*.sh
// RUN: FileCheck --check-prefix=CHECKYAML %s -input-file \
// RUN:   %t/crash-vfs-*.cache/vfs/vfs.yaml
// RUN: find %t/crash-vfs-*.cache/vfs | \
// RUN:   grep "%t/actual_module2.map" | count 1

#include <stdio.h>

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}.m
// CHECK-NEXT: note: diagnostic msg: {{.*}}.cache

// CHECKSH: # Crash reproducer
// CHECKSH-NEXT: # Driver args: "-fsyntax-only"
// CHECKSH-NEXT: # Original command: {{.*$}}
// CHECKSH-NEXT: "-cc1"
// CHECKSH: "crash-vfs-{{[^ ]*}}.m"
// CHECKSH: "-ivfsoverlay" "crash-vfs-{{[^ ]*}}.cache/vfs/vfs.yaml"
// CHECKSH: "-fmodules-cache-path=crash-vfs-{{[^ ]*}}.cache/repro-modules"

// CHECKYAML: 'case-sensitive':
// CHECKYAML-NEXT: 'use-external-names': 'false',
// CHECKYAML-NEXT: 'overlay-relative': 'true',
// CHECKYAML-NEXT: 'ignore-non-existent-contents': 'false'
// CHECKYAML: 'type': 'directory'
// CHECKYAML: 'name': "/[[PATH:.*]]/example"
// CHECKYAML: 'contents': [
// CHECKYAML-NEXT:   {
// CHECKYAML-NEXT:     'type': 'file',
// CHECKYAML-NEXT:     'name': "module.map",
// CHECKYAML-NEXT:     'external-contents': "/[[OTHERPATH:.*]]/actual_module2.map"
