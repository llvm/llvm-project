// REQUIRES: x86-registered-target
// REQUIRES: arm-registered-target
// REQUIRES: aarch64-registered-target
// REQUIRES: riscv-registered-target

// RUN: rm -rf %t

// Sanity check one of the compilations.
// RUN: %clang_cc1 -triple aarch64-unknown-unknown -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I %S/Inputs -fsyntax-only %s -verify -DSANITY_CHECK
// expected-no-diagnostics

// Check all the targets:
// RUN: not %clang_cc1 -triple armv7-unknown-unknown -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I %S/Inputs -fsyntax-only  %s 2> %t.aarch32
// RUN: FileCheck %s -check-prefix=AARCH32 < %t.aarch32
// RUN: not %clang_cc1 -triple aarch64-unknown-unknown -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I %S/Inputs -fsyntax-only  %s 2> %t.aarch64
// RUN: FileCheck %s -check-prefix=AARCH64 < %t.aarch64
// RUN: not %clang_cc1 -triple i386-unknown-unknown -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I %S/Inputs -fsyntax-only  %s 2> %t.x86_32
// RUN: FileCheck %s -check-prefix=X86_32 < %t.x86_32
// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I %S/Inputs -fsyntax-only  %s 2> %t.x86_64
// RUN: FileCheck %s -check-prefix=X86_64 < %t.x86_64
// RUN: not %clang_cc1 -triple riscv32-unknown-unknown -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I %S/Inputs -fsyntax-only  %s 2> %t.riscv32
// RUN: FileCheck %s -check-prefix=RISCV32 < %t.riscv32
// RUN: not %clang_cc1 -triple riscv64-unknown-unknown -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I %S/Inputs -fsyntax-only  %s 2> %t.riscv64
// RUN: FileCheck %s -check-prefix=RISCV64 < %t.riscv64

#ifndef SANITY_CHECK
@import TargetFeatures;
// AARCH32-NOT: module 'TargetFeatures' requires
// AARCH64-NOT: module 'TargetFeatures' requires
// X86_32-NOT: module 'TargetFeatures' requires
// X86_64-NOT: module 'TargetFeatures' requires
// RISCV32-NOT: module 'TargetFeatures' requires
// RISCV64-NOT: module 'TargetFeatures' requires
@import TargetFeatures.arm;
// AARCH32-NOT: module 'TargetFeatures.arm' requires
// AARCH64-NOT: module 'TargetFeatures.arm' requires
// X86_32: module 'TargetFeatures.arm' requires feature 'arm'
// X86_64: module 'TargetFeatures.arm' requires feature 'arm'
// RISCV32: module 'TargetFeatures.arm' requires
// RISCV64: module 'TargetFeatures.arm' requires
@import TargetFeatures.arm.aarch32;
// AARCH32-NOT: module 'TargetFeatures.arm.aarch32' requires
// AARCH64: module 'TargetFeatures.arm.aarch32' requires feature 'aarch32'
// X86_32: module 'TargetFeatures.arm.aarch32' requires feature 
// X86_64: module 'TargetFeatures.arm.aarch32' requires feature
// RISCV32: module 'TargetFeatures.arm.aarch32' requires feature
// RISCV64: module 'TargetFeatures.arm.aarch32' requires feature
#endif

@import TargetFeatures.arm.aarch64;
// AARCH32: module 'TargetFeatures.arm.aarch64' requires feature 'aarch64'
// AARCH64-NOT: module 'TargetFeatures.arm.aarch64' requires
// X86_32: module 'TargetFeatures.arm.aarch64' requires feature 
// X86_64: module 'TargetFeatures.arm.aarch64' requires feature
// RISCV32: module 'TargetFeatures.arm.aarch64' requires feature
// RISCV64: module 'TargetFeatures.arm.aarch64' requires feature

#ifndef SANITY_CHECK
@import TargetFeatures.x86;
// AARCH32:  module 'TargetFeatures.x86' requires feature 'x86'
// AARCH64:  module 'TargetFeatures.x86' requires feature 'x86'
// X86_32-NOT: module 'TargetFeatures.x86' requires
// X86_64-NOT: module 'TargetFeatures.x86' requires
// RISCV32:  module 'TargetFeatures.x86' requires feature 'x86'
// RISCV64:  module 'TargetFeatures.x86' requires feature 'x86'
@import TargetFeatures.x86.x86_32;
// AARCH32:  module 'TargetFeatures.x86.x86_32' requires feature
// AARCH64:  module 'TargetFeatures.x86.x86_32' requires feature
// X86_32-NOT: module 'TargetFeatures.x86.x86_32' requires
// X86_64: module 'TargetFeatures.x86.x86_32' requires feature 'x86_32'
// RISCV32:  module 'TargetFeatures.x86.x86_32' requires feature
// RISCV64:  module 'TargetFeatures.x86.x86_32' requires feature
@import TargetFeatures.x86.x86_64;
// AARCH32:  module 'TargetFeatures.x86.x86_64' requires feature
// AARCH64:  module 'TargetFeatures.x86.x86_64' requires feature
// X86_32: module 'TargetFeatures.x86.x86_64' requires feature 'x86_64'
// X86_64-NOT: module 'TargetFeatures.x86.x86_64' requires
// RISCV32:  module 'TargetFeatures.x86.x86_64' requires feature
// RISCV64:  module 'TargetFeatures.x86.x86_64' requires feature
@import TargetFeatures.riscv;
// AARCH32:  module 'TargetFeatures.riscv' requires feature
// AARCH64:  module 'TargetFeatures.riscv' requires feature
// X86_32: module 'TargetFeatures.riscv' requires feature
// X86_64: module 'TargetFeatures.riscv' requires feature
// RISCV32-NOT: module 'TargetFeatures.riscv' requires feature
// RISCV64-NOT: module 'TargetFeatures.riscv' requires feature
@import TargetFeatures.riscv.riscv32;
// AARCH32:  module 'TargetFeatures.riscv.riscv32' requires feature
// AARCH64:  module 'TargetFeatures.riscv.riscv32' requires feature
// X86_32: module 'TargetFeatures.riscv.riscv32' requires feature
// X86_64: module 'TargetFeatures.riscv.riscv32' requires feature
// RISCV32-NOT: module 'TargetFeatures.riscv.riscv32' requires feature
// RISCV64: module 'TargetFeatures.riscv.riscv32' requires feature 'riscv32'
@import TargetFeatures.riscv.riscv64;
// AARCH32:  module 'TargetFeatures.riscv.riscv64' requires feature
// AARCH64:  module 'TargetFeatures.riscv.riscv64' requires feature
// X86_32: module 'TargetFeatures.riscv.riscv64' requires feature
// X86_64: module 'TargetFeatures.riscv.riscv64' requires feature
// RISCV32: module 'TargetFeatures.riscv.riscv64' requires feature 'riscv64'
// RISCV64-NOT: module 'TargetFeatures.riscv.riscv64' requires feature
#endif
