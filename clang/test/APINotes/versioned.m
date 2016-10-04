// RUN: rm -rf %t && mkdir -p %t

// Build and check the unversioned module file.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Unversioned -fdisable-module-hash -fapinotes-modules -fapinotes-cache-path=%t/APINotesCache -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s
// RUN: %clang_cc1 -ast-print %t/ModulesCache/Unversioned/VersionedKit.pcm | FileCheck -check-prefix=CHECK-UNVERSIONED %s

// Build and check the versioned module file.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Versioned -fdisable-module-hash -fapinotes-modules -fapinotes-cache-path=%t/APINotesCache -fapinotes-swift-version=3.0 -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s
// RUN: %clang_cc1 -ast-print %t/ModulesCache/Versioned/VersionedKit.pcm | FileCheck -check-prefix=CHECK-VERSIONED %s

#import <VersionedKit/VersionedKit.h>

// CHECK-UNVERSIONED: void moveToPoint(double x, double y) __attribute__((swift_name("moveTo(x:y:)")));
// CHECK-VERSIONED: void moveToPoint(double x, double y) __attribute__((swift_name("moveTo(a:b:)")));

