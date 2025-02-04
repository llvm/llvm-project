// REQUIRES: shell, amdclang
// UNSUPPORTED: system-windows
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: ln -s amdclang %t/amdfoo
// RUN: not %t/amdfoo 2>&1 | FileCheck %s --check-prefix=DOES_NOT_EXIST
// RUN: ln -s amdclang %t/foo
// RUN: not %t/foo 2>&1 | FileCheck %s --check-prefix=BAD_PREFIX
//
// DOES_NOT_EXIST: binary '{{.*}}' does not exist
// BAD_PREFIX: binary '{{.*}}' not prefixed by 'amd'
