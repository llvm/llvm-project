// REQUIRES: crash-recovery, shell

// FIXME: This XFAIL is cargo-culted from crash-report.c. Do we need it?
// XFAIL: mingw32

// Test that clang is capable of collecting the right header files in the
// crash reproducer if there's a symbolic link component in the path.

// RUN: rm -rf %t
// RUN: mkdir -p %t/i %t/m %t %t/sysroot
// RUN: cp -a %S/Inputs/crash-recovery/usr %t/i/
// RUN: ln -s include/tcl-private %t/i/usr/x

// RUN: not env FORCE_CLANG_DIAGNOSTICS_CRASH= TMPDIR=%t TEMP=%t TMP=%t \
// RUN: %clang -fsyntax-only %s -I %/t/i -isysroot %/t/sysroot/ \
// RUN:     -fmodules -fmodules-cache-path=%t/m/ 2>&1 | FileCheck %s

// RUN: FileCheck --check-prefix=CHECKSRC %s -input-file %t/crash-vfs-*.m
// RUN: FileCheck --check-prefix=CHECKSH %s -input-file %t/crash-vfs-*.sh
// RUN: FileCheck --check-prefix=CHECKYAML %s -input-file \
// RUN: %t/crash-vfs-*.cache/vfs/vfs.yaml
// RUN: find %t/crash-vfs-*.cache/vfs | \
// RUN:   grep "usr/include/stdio.h" | count 1

#include "usr/x/../stdio.h"

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}.m
// CHECK-NEXT: note: diagnostic msg: {{.*}}.cache

// CHECKSRC: @import cstd.stdio;

// CHECKSH: # Crash reproducer
// CHECKSH-NEXT: # Driver args: "-fsyntax-only"
// CHECKSH-NEXT: # Original command: {{.*$}}
// CHECKSH-NEXT: "-cc1"
// CHECKSH: "-isysroot" "{{[^"]*}}/sysroot/"
// CHECKSH-NOT: "-fmodules-cache-path="
// CHECKSH: "crash-vfs-{{[^ ]*}}.m"
// CHECKSH: "-ivfsoverlay" "crash-vfs-{{[^ ]*}}.cache/vfs/vfs.yaml"
// CHECKSH: "-fmodules-cache-path=crash-vfs-{{[^ ]*}}.cache/modules"

// CHECKYAML: 'case-sensitive':
// CHECKYAML-NEXT: 'use-external-names': 'false',
// CHECKYAML-NEXT: 'overlay-relative': 'true',

// CHECKYAML: 'type': 'directory'
// CHECKYAML: 'name': "/[[PATH:.*]]/i/usr",
// CHECKYAML-NEXT: 'contents': [
// CHECKYAML-NEXT:   {
// CHECKYAML-NEXT:     'type': 'file',
// CHECKYAML-NEXT:     'name': "module.map",
// CHECKYAML-NEXT:     'external-contents': "/[[PATH]]/i/usr/include/module.map"
// CHECKYAML-NEXT:   },

// Test that by using the previous generated YAML file clang is able to find the
// right files inside the overlay and map the virtual request for a path that
// previously contained a symlink to work. To make sure of this, wipe out the
// %/t/i directory containing the symlink component.

// RUN: rm -rf %/t/i
// RUN: unset FORCE_CLANG_DIAGNOSTICS_CRASH
// RUN: %clang -E %s -I %/t/i -isysroot %/t/sysroot/ \
// RUN:     -ivfsoverlay %t/crash-vfs-*.cache/vfs/vfs.yaml -fmodules \
// RUN:     -fmodules-cache-path=%t/m/ 2>&1 \
// RUN:     | FileCheck %s --check-prefix=CHECKOVERLAY

// CHECKOVERLAY: @import cstd.stdio; /* clang -E: implicit import for "/{{[^ ].*}}/i/usr/x/../stdio.h" */
