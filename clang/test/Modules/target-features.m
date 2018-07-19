// REQUIRES: x86-registered-target
// REQUIRES: arm-registered-target
// REQUIRES: aarch64-registered-target

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

#ifndef SANITY_CHECK
@import TargetFeatures;
// AARCH32-NOT: module 'TargetFeatures' requires
// AARCH64-NOT: module 'TargetFeatures' requires
// X86_32-NOT: module 'TargetFeatures' requires
// X86_64-NOT: module 'TargetFeatures' requires
@import TargetFeatures.arm;
// AARCH32-NOT: module 'TargetFeatures.arm' requires
// AARCH64-NOT: module 'TargetFeatures.arm' requires
// X86_32: module 'TargetFeatures.arm' requires feature 'arm'
// X86_64: module 'TargetFeatures.arm' requires feature 'arm'
@import TargetFeatures.arm.aarch32;
// AARCH32-NOT: module 'TargetFeatures.arm.aarch32' requires
// AARCH64: module 'TargetFeatures.arm.aarch32' requires feature 'aarch32'
// X86_32: module 'TargetFeatures.arm.aarch32' requires feature 
// X86_64: module 'TargetFeatures.arm.aarch32' requires feature
#endif

@import TargetFeatures.arm.aarch64;
// AARCH32: module 'TargetFeatures.arm.aarch64' requires feature 'aarch64'
// AARCH64-NOT: module 'TargetFeatures.arm.aarch64' requires
// X86_32: module 'TargetFeatures.arm.aarch64' requires feature 
// X86_64: module 'TargetFeatures.arm.aarch64' requires feature

#ifndef SANITY_CHECK
@import TargetFeatures.x86;
// AARCH32:  module 'TargetFeatures.x86' requires feature 'x86'
// AARCH64:  module 'TargetFeatures.x86' requires feature 'x86'
// X86_32-NOT: module 'TargetFeatures.x86' requires
// X86_64-NOT: module 'TargetFeatures.x86' requires
@import TargetFeatures.x86.x86_32;
// AARCH32:  module 'TargetFeatures.x86.x86_32' requires feature
// AARCH64:  module 'TargetFeatures.x86.x86_32' requires feature
// X86_32-NOT: module 'TargetFeatures.x86.x86_32' requires
// X86_64: module 'TargetFeatures.x86.x86_32' requires feature 'x86_32'
@import TargetFeatures.x86.x86_64;
// AARCH32:  module 'TargetFeatures.x86.x86_64' requires feature
// AARCH64:  module 'TargetFeatures.x86.x86_64' requires feature
// X86_32: module 'TargetFeatures.x86.x86_64' requires feature 'x86_64'
// X86_64-NOT: module 'TargetFeatures.x86.x86_64' requires
#endif
