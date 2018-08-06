// REQUIRES: crash-recovery, shell

// FIXME: This XFAIL is cargo-culted from crash-report.c. Do we need it?
// XFAIL: mingw32

// RUN: rm -rf %t
// RUN: mkdir -p %t/i %t/m %t
// RUN: cp -R %S/Inputs/crash-recovery/Frameworks %t/i/
// RUN: mkdir -p %t/i/Frameworks/A.framework/Frameworks
// RUN: ln -s ../../B.framework %t/i/Frameworks/A.framework/Frameworks/B.framework

// RUN: not env FORCE_CLANG_DIAGNOSTICS_CRASH= TMPDIR=%t TEMP=%t TMP=%t \
// RUN: %clang -nostdinc -fsyntax-only %s \
// RUN:     -F %/t/i/Frameworks -fmodules \
// RUN:     -fmodules-cache-path=%t/m/ 2>&1 | FileCheck %s

// RUN: FileCheck --check-prefix=CHECKYAML %s -input-file \
// RUN:         %t/crash-vfs-*.cache/vfs/vfs.yaml
// RUN: find %t/crash-vfs-*.cache/vfs | \
// RUN:   grep "B.framework/Headers/B.h" | count 1

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}.m
// CHECK-NEXT: note: diagnostic msg: {{.*}}.cache

// CHECKYAML:      'type': 'directory',
// CHECKYAML:      'name': "/[[PATH:.*]]/i/Frameworks/A.framework/Frameworks/B.framework/Headers",
// CHECKYAML-NEXT:      'contents': [
// CHECKYAML-NEXT:        {
// CHECKYAML-NEXT:          'type': 'file',
// CHECKYAML-NEXT:          'name': "B.h",
// CHECKYAML-NEXT:          'external-contents': "/[[PATH]]/i/Frameworks/B.framework/Headers/B.h"

// CHECKYAML:      'type': 'directory',
// CHECKYAML:      'name': "/[[PATH]]/i/Frameworks/B.framework/Headers",
// CHECKYAML-NEXT:      'contents': [
// CHECKYAML-NEXT:        {
// CHECKYAML-NEXT:          'type': 'file',
// CHECKYAML-NEXT:          'name': "B.h",
// CHECKYAML-NEXT:          'external-contents': "/[[PATH]]/i/Frameworks/B.framework/Headers/B.h"

@import I;

// Run the reproducer script - regular exit code is enough to test it works. The
// intent here is to guarantee that the collect umbrella headers into the VFS
// can be used, testing that vfs::recursive_directory_iterator is used correctly
// Make sure to erase the include paths used to build the modules to guarantee
// that the VFS overlay won't fallback to use it. Also wipe out the module cache
// to force header search.
//
// RUN: cd %t
// RUN: rm -rf i
// RUN: rm -rf crash-vfs-umbrella-*.cache/modules/*
// RUN: chmod 755 crash-vfs-*.sh
// RUN: ./crash-vfs-*.sh
