// REQUIRES: x86-registered-target

//--- Invocation `clang --config x86_64-qqq -m32` loads `i386-qqq.cfg` if the latter exists.
//
// RUN: %clang --config-system-dir=%S/Inputs/config --config-user-dir= --config x86_64-qqq -m32 -c %s -### 2>&1 | FileCheck %s -check-prefix CHECK-RELOAD
// CHECK-RELOAD: Target: i386
// CHECK-RELOAD: Configuration file: {{.*}}Inputs{{.}}config{{.}}i386-qqq.cfg


//--- Invocation `clang --config x86_64-qqq2 -m32` loads `i386.cfg` if the latter exists in another search directory.
//
// RUN: %clang --config-system-dir=%S/Inputs/config --config-user-dir=%S/Inputs/config2 --config x86_64-qqq2 -m32 -c %s -### 2>&1 | FileCheck %s -check-prefix CHECK-RELOAD1
// CHECK-RELOAD1: Target: i386
// CHECK-RELOAD1: Configuration file: {{.*}}Inputs{{.}}config2{{.}}i386.cfg


//--- Invocation `clang --config x86_64-qqq2 -m32` loads `x86_64-qqq2.cfg` if `i386-qqq2.cfg` and `i386.cfg` do not exist.
//
// RUN: %clang --config-system-dir=%S/Inputs/config --config-user-dir= --config x86_64-qqq2 -m32 -c %s -### 2>&1 | FileCheck %s -check-prefix CHECK-RELOAD2
// note: target is overriden due to -m32
// CHECK-RELOAD2: Target: i386
// CHECK-RELOAD2: Configuration file: {{.*}}Inputs{{.}}config{{.}}x86_64-qqq2.cfg


//--- Invocation `clang --config i386-qqq3 -m64` loads `x86_64.cfg` if `x86_64-qqq3.cfg` does not exist.
//
// RUN: %clang --config-system-dir=%S/Inputs/config --config-user-dir= --config i386-qqq3 -m64 -c %s -### 2>&1 | FileCheck %s -check-prefix CHECK-RELOAD3
// CHECK-RELOAD3: Target: x86_64
// CHECK-RELOAD3: Configuration file: {{.*}}Inputs{{.}}config{{.}}x86_64.cfg


//--- Invocation `clang --config x86_64-qqq -target i386` loads `i386-qqq.cfg` if the latter exists.
//
// RUN: %clang --config-system-dir=%S/Inputs/config --config-user-dir= --config x86_64-qqq -target i386 -c %s -### 2>&1 | FileCheck %s -check-prefix CHECK-RELOAD4
// CHECK-RELOAD4: Target: i386
// CHECK-RELOAD4: Configuration file: {{.*}}Inputs{{.}}config{{.}}i386-qqq.cfg


//--- Invocation `clang --config x86_64-qqq2 -target i386` loads `x86_64-qqq2.cfg` if `i386-qqq2.cfg` and `i386.cfg` do not exist.
//
// RUN: %clang --config-system-dir=%S/Inputs/config --config-user-dir= --config x86_64-qqq2 -target i386 -c %s -### 2>&1 | FileCheck %s -check-prefix CHECK-RELOAD5
// note: target is overriden due to -target i386
// CHECK-RELOAD5: Target: i386
// CHECK-RELOAD5: Configuration file: {{.*}}Inputs{{.}}config{{.}}x86_64-qqq2.cfg


//--- Invocation `clang --config x86_64-qqq -target i386 -m64` loads `x86_64-qqq.cfg`.
//
// RUN: %clang --config-system-dir=%S/Inputs/config --config-user-dir= --config x86_64-qqq -target i386 -m64 -c %s -### 2>&1 | FileCheck %s -check-prefix CHECK-RELOAD6
// CHECK-RELOAD6: Target: x86_64
// CHECK-RELOAD6: Configuration file: {{.*}}Inputs{{.}}config{{.}}x86_64-qqq.cfg
