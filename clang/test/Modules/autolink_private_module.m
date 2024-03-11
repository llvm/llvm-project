// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}

// Test that autolink hints for frameworks don't use the private module name.
// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -emit-llvm -o - -fmodules-cache-path=%t/ModuleCache -fmodules -fimplicit-module-maps -F %t/Frameworks %t/test.m | FileCheck %s

// CHECK:     !{!"-framework", !"Autolink"}
// CHECK-NOT: !{!"-framework", !"Autolink_Private"}

//--- test.m
#include <Autolink/Autolink.h>
#include <Autolink/Autolink_Private.h>

//--- Frameworks/Autolink.framework/Headers/Autolink.h
void public();

//--- Frameworks/Autolink.framework/PrivateHeaders/Autolink_Private.h
void private();

//--- Frameworks/Autolink.framework/Modules/module.modulemap
framework module Autolink { header "Autolink.h"}

//--- Frameworks/Autolink.framework/Modules/module.private.modulemap
framework module Autolink_Private { header "Autolink_Private.h"}

