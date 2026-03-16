// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: clang-scan-deps -format experimental-include-tree-full -cas-path %t/cas -o %t/deps.json -- \
// RUN:     %clang -c %t/tu.c -o %t/tu.o -F %t/frameworks -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache

// RUN: %deps-to-rsp %t/deps.json --module-name A > %t/mod_A.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name A_Private > %t/mod_AP.rsp
// RUN: %clang @%t/mod_A.rsp -o %t/A.pcm
// RUN: %clang @%t/mod_AP.rsp -o %t/AP.pcm

// RUN: cat %t/mod_A.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/A.casid
// RUN: cat %t/mod_AP.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/AP.casid

/// Check the serialization.
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/A.casid | FileCheck %s --check-prefix PUBLIC
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/AP.casid | FileCheck %s --check-prefix PRIVATE

// PUBLIC: A (framework)
// PUBLIC-NOT: (private)
// PRIVATE: A_Private (framework) (private)

//--- frameworks/A.framework/Modules/module.modulemap
framework module A { header "A.h" export *}

//--- frameworks/A.framework/Modules/module.private.modulemap
framework module A_Private { header "A_Private.h" export *}

//--- frameworks/A.framework/Headers/A.h
struct A {
  int x;
};

//--- frameworks/A.framework/PrivateHeaders/A_Private.h
#include "A/A.h"
struct AP {
  int x;
};

//--- tu.c
#import "A/A_Private.h"
