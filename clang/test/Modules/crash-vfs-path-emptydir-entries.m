// UNSUPPORTED: system-windows
// REQUIRES: crash-recovery

// FIXME: This XFAIL is cargo-culted from crash-report.c. Do we need it?
// XFAIL: target={{.*-windows-gnu}}

// Test clang can collect symbolic link headers used in modules.
// crash reproducer if there's a symbolic link header file used in a module.

// RUN: rm -rf %t
// RUN: mkdir -p %t/i %t/m %t %t/sysroot
// RUN: cp -R %S/Inputs/crash-recovery/usr %t/i/

// RUN: env FORCE_CLANG_DIAGNOSTICS_CRASH= TMPDIR=%t TEMP=%t TMP=%t \
// RUN: not %clang -fsyntax-only %s -I %/t/i -isysroot %/t/sysroot/ \
// RUN:     -fmodules -fmodules-cache-path=%t/m/ 2>&1 | FileCheck %s

// RUN: FileCheck --check-prefix=CHECKSRC %s -input-file %t/crash-vfs-*.m
// RUN: FileCheck --check-prefix=CHECKSH %s -input-file %t/crash-vfs-*.sh
// RUN: FileCheck --check-prefix=CHECKYAML %s -input-file \
// RUN: %t/crash-vfs-*.cache/vfs/vfs.yaml
// RUN: find %t/crash-vfs-*.cache/vfs | \
// RUN:   grep "usr/include/stdio.h" | count 1

#include "usr/include/stdio.h"

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}.m
// CHECK-NEXT: note: diagnostic msg: {{.*}}.cache

// CHECKSRC: #pragma clang module import cstd.stdio

// CHECKSH: # Crash reproducer
// CHECKSH-NEXT: # Driver args: "-fsyntax-only"
// CHECKSH-NEXT: # Original command: {{.*$}}
// CHECKSH-NEXT: "-cc1"
// CHECKSH: "-isysroot" "{{[^"]*}}/sysroot/"
// CHECKSH-NOT: "-fmodules-cache-path="
// CHECKSH: "crash-vfs-{{[^ ]*}}.m"
// CHECKSH: "-ivfsoverlay" "crash-vfs-{{[^ ]*}}.cache/vfs/vfs.yaml"
// CHECKSH: "-fmodules-cache-path=crash-vfs-{{[^ ]*}}.cache/repro-modules"

// CHECKYAML: 'type': 'directory',
// CHECKYAML: 'name': "",
// CHECKYAML-NEXT: 'contents': [
// CHECKYAML-NEXT:   {
// CHECKYAML-NEXT:     'type': 'file',
// CHECKYAML-NEXT:     'name': "pthread_impl.h",
// CHECKYAML-NEXT:     'external-contents': "/{{.*}}/i/usr/include/pthread_impl.h"
// CHECKYAML-NEXT:   },
