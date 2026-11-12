// RUN: rm -rf %t && mkdir -p %t

// Build and check the unversioned module file.
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Unversioned -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s
// RUN: %clang_cc1 -ast-print %t/ModulesCache/Unversioned/VersionedKit.pcm | FileCheck -check-prefix=CHECK-UNVERSIONED %s

// Build and check the various versions.
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Versioned3 -fdisable-module-hash -fapinotes-modules -fapinotes-swift-version=3 -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s
// RUN: %clang_cc1 -ast-print %t/ModulesCache/Versioned3/VersionedKit.pcm | FileCheck -check-prefix=CHECK-VERSIONED-3 %s

// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Versioned4 -fdisable-module-hash -fapinotes-modules -fapinotes-swift-version=4 -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s
// RUN: %clang_cc1 -ast-print %t/ModulesCache/Versioned4/VersionedKit.pcm | FileCheck -check-prefix=CHECK-VERSIONED-4 %s

// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Versioned5 -fdisable-module-hash -fapinotes-modules -fapinotes-swift-version=5 -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s
// RUN: %clang_cc1 -ast-print %t/ModulesCache/Versioned5/VersionedKit.pcm | FileCheck -check-prefix=CHECK-VERSIONED-5 %s

#import <VersionedKit/VersionedKit.h>

// CHECK-UNVERSIONED: typedef int MultiVersionedTypedef4;
// CHECK-UNVERSIONED: typedef int MultiVersionedTypedef4Notes __attribute__((swift_name("MultiVersionedTypedef4Notes_NEW")));
// CHECK-UNVERSIONED: typedef int MultiVersionedTypedef4Header __attribute__((swift_name("MultiVersionedTypedef4Header_NEW")));
// CHECK-UNVERSIONED: typedef int MultiVersionedTypedef34;
// CHECK-UNVERSIONED: typedef int MultiVersionedTypedef34Notes __attribute__((swift_name("MultiVersionedTypedef34Notes_NEW")));
// CHECK-UNVERSIONED: typedef int MultiVersionedTypedef34Header __attribute__((swift_name("MultiVersionedTypedef34Header_NEW")));
// CHECK-UNVERSIONED: typedef int MultiVersionedTypedef45;
// CHECK-UNVERSIONED: typedef int MultiVersionedTypedef45Notes __attribute__((swift_name("MultiVersionedTypedef45Notes_NEW")));
// CHECK-UNVERSIONED: typedef int MultiVersionedTypedef45Header __attribute__((swift_name("MultiVersionedTypedef45Header_NEW")));
// CHECK-UNVERSIONED: typedef int MultiVersionedTypedef345;
// CHECK-UNVERSIONED: typedef int MultiVersionedTypedef345Notes __attribute__((swift_name("MultiVersionedTypedef345Notes_NEW")));
// CHECK-UNVERSIONED: typedef int MultiVersionedTypedef345Header __attribute__((swift_name("MultiVersionedTypedef345Header_NEW")));

// CHECK-VERSIONED-3: typedef int MultiVersionedTypedef4 __attribute__((swift_name("MultiVersionedTypedef4_4")));
// CHECK-VERSIONED-3: typedef int MultiVersionedTypedef4Notes __attribute__((swift_name("MultiVersionedTypedef4Notes_4")));
// CHECK-VERSIONED-3: typedef int MultiVersionedTypedef4Header __attribute__((swift_name("MultiVersionedTypedef4Header_4")));
// CHECK-VERSIONED-3: typedef int MultiVersionedTypedef34 __attribute__((swift_name("MultiVersionedTypedef34_3")));
// CHECK-VERSIONED-3: typedef int MultiVersionedTypedef34Notes __attribute__((swift_name("MultiVersionedTypedef34Notes_3")));
// CHECK-VERSIONED-3: typedef int MultiVersionedTypedef34Header __attribute__((swift_name("MultiVersionedTypedef34Header_3")));
// CHECK-VERSIONED-3: typedef int MultiVersionedTypedef45 __attribute__((swift_name("MultiVersionedTypedef45_4")));
// CHECK-VERSIONED-3: typedef int MultiVersionedTypedef45Notes __attribute__((swift_name("MultiVersionedTypedef45Notes_4")));
// CHECK-VERSIONED-3: typedef int MultiVersionedTypedef45Header __attribute__((swift_name("MultiVersionedTypedef45Header_4")));
// CHECK-VERSIONED-3: typedef int MultiVersionedTypedef345 __attribute__((swift_name("MultiVersionedTypedef345_3")));
// CHECK-VERSIONED-3: typedef int MultiVersionedTypedef345Notes __attribute__((swift_name("MultiVersionedTypedef345Notes_3")));
// CHECK-VERSIONED-3: typedef int MultiVersionedTypedef345Header __attribute__((swift_name("MultiVersionedTypedef345Header_3")));

// CHECK-VERSIONED-4: typedef int MultiVersionedTypedef4 __attribute__((swift_name("MultiVersionedTypedef4_4")));
// CHECK-VERSIONED-4: typedef int MultiVersionedTypedef4Notes __attribute__((swift_name("MultiVersionedTypedef4Notes_4")));
// CHECK-VERSIONED-4: typedef int MultiVersionedTypedef4Header __attribute__((swift_name("MultiVersionedTypedef4Header_4")));
// CHECK-VERSIONED-4: typedef int MultiVersionedTypedef34 __attribute__((swift_name("MultiVersionedTypedef34_4")));
// CHECK-VERSIONED-4: typedef int MultiVersionedTypedef34Notes __attribute__((swift_name("MultiVersionedTypedef34Notes_4")));
// CHECK-VERSIONED-4: typedef int MultiVersionedTypedef34Header __attribute__((swift_name("MultiVersionedTypedef34Header_4")));
// CHECK-VERSIONED-4: typedef int MultiVersionedTypedef45 __attribute__((swift_name("MultiVersionedTypedef45_4")));
// CHECK-VERSIONED-4: typedef int MultiVersionedTypedef45Notes __attribute__((swift_name("MultiVersionedTypedef45Notes_4")));
// CHECK-VERSIONED-4: typedef int MultiVersionedTypedef45Header __attribute__((swift_name("MultiVersionedTypedef45Header_4")));
// CHECK-VERSIONED-4: typedef int MultiVersionedTypedef345 __attribute__((swift_name("MultiVersionedTypedef345_4")));
// CHECK-VERSIONED-4: typedef int MultiVersionedTypedef345Notes __attribute__((swift_name("MultiVersionedTypedef345Notes_4")));
// CHECK-VERSIONED-4: typedef int MultiVersionedTypedef345Header __attribute__((swift_name("MultiVersionedTypedef345Header_4")));

// CHECK-VERSIONED-5: typedef int MultiVersionedTypedef4;
// CHECK-VERSIONED-5: typedef int MultiVersionedTypedef4Notes __attribute__((swift_name("MultiVersionedTypedef4Notes_NEW")));
// CHECK-VERSIONED-5: typedef int MultiVersionedTypedef4Header __attribute__((swift_name("MultiVersionedTypedef4Header_NEW")));
// CHECK-VERSIONED-5: typedef int MultiVersionedTypedef34;
// CHECK-VERSIONED-5: typedef int MultiVersionedTypedef34Notes __attribute__((swift_name("MultiVersionedTypedef34Notes_NEW")));
// CHECK-VERSIONED-5: typedef int MultiVersionedTypedef34Header __attribute__((swift_name("MultiVersionedTypedef34Header_NEW")));
// CHECK-VERSIONED-5: typedef int MultiVersionedTypedef45 __attribute__((swift_name("MultiVersionedTypedef45_5")));
// CHECK-VERSIONED-5: typedef int MultiVersionedTypedef45Notes __attribute__((swift_name("MultiVersionedTypedef45Notes_5")));
// CHECK-VERSIONED-5: typedef int MultiVersionedTypedef45Header __attribute__((swift_name("MultiVersionedTypedef45Header_5")));
// CHECK-VERSIONED-5: typedef int MultiVersionedTypedef345 __attribute__((swift_name("MultiVersionedTypedef345_5")));
// CHECK-VERSIONED-5: typedef int MultiVersionedTypedef345Notes __attribute__((swift_name("MultiVersionedTypedef345Notes_5")));
// CHECK-VERSIONED-5: typedef int MultiVersionedTypedef345Header __attribute__((swift_name("MultiVersionedTypedef345Header_5")));
