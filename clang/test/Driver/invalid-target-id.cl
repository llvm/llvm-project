// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: not %clang -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx908xnack -nostdlib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=NOPLUS %s

// NOPLUS: error: invalid target ID 'gfx908xnack'

// RUN: not %clang -target amdgcn-amd-amdpal \
// RUN:   -mcpu=gfx908:xnack+:xnack+ -nostdlib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=ORDER %s

// ORDER: error: invalid target ID 'gfx908:xnack+:xnack+'

// RUN: not %clang -target amdgcn--mesa3d \
// RUN:   -mcpu=gfx908:unknown+ -nostdlib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=UNK %s

// UNK: error: invalid target ID 'gfx908:unknown+'

// RUN: not %clang -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx908:sramecc+:unknown+ -nostdlib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=MIXED %s

// MIXED: error: invalid target ID 'gfx908:sramecc+:unknown+'

// RUN: not %clang -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx900:sramecc+ -nostdlib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=UNSUP %s

// UNSUP: error: invalid target ID 'gfx900:sramecc+'

// RUN: not %clang -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx900:xnack -nostdlib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=NOSIGN %s

// NOSIGN: error: invalid target ID 'gfx900:xnack'

// RUN: not %clang -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx900+xnack -nostdlib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=NOCOLON %s

// NOCOLON: error: invalid target ID 'gfx900+xnack'
