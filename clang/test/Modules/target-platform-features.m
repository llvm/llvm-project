// Clear and create directories
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: mkdir %t/cache
// RUN: mkdir %t/InputsA

// RUN: echo "module RequiresMacOS {"       >> %t/InputsA/module.map
// RUN: echo "  requires macos"             >> %t/InputsA/module.map
// RUN: echo "}"                            >> %t/InputsA/module.map
// RUN: echo "module RequiresNotiOS {"      >> %t/InputsA/module.map
// RUN: echo "  requires !ios"              >> %t/InputsA/module.map
// RUN: echo "}"                            >> %t/InputsA/module.map
// RUN: echo "module RequiresMain {"        >> %t/InputsA/module.map
// RUN: echo "  module SubRequiresNotiOS {" >> %t/InputsA/module.map
// RUN: echo "    requires !ios"            >> %t/InputsA/module.map
// RUN: echo "  }"                          >> %t/InputsA/module.map
// RUN: echo "}"                            >> %t/InputsA/module.map

// RUN: %clang_cc1 -triple=x86_64-apple-macosx10.6 -DENABLE_DARWIN -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -x objective-c -I%t/InputsA -verify %s 
// expected-no-diagnostics

// RUN: not %clang_cc1 -triple=arm64-apple-ios -DENABLE_DARWIN -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -x objective-c -I%t/InputsA %s  2> %t/notios
// RUN: FileCheck %s -check-prefix=CHECK-IOS < %t/notios
#ifdef ENABLE_DARWIN
// CHECK-IOS: module 'RequiresMacOS' requires feature 'macos'
@import RequiresMacOS;
// CHECK-IOS: module 'RequiresNotiOS' is incompatible with feature 'ios'
@import RequiresNotiOS;
// We should never get errors for submodules that don't match
// CHECK-IOS-NOT: module 'RequiresMain'
@import RequiresMain;
#endif

// RUN: mkdir %t/InputsB
// RUN: echo "module RequiresiOSSim {"     >> %t/InputsB/module.map
// RUN: echo "  requires iossimulator"      >> %t/InputsB/module.map
// RUN: echo "}"                            >> %t/InputsB/module.map
// RUN: %clang_cc1 -triple=x86_64-apple-iossimulator -DENABLE_IOSSIM -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -x objective-c -I%t/InputsB %s  -verify
// RUN: %clang_cc1 -triple=x86_64-apple-ios-simulator -DENABLE_IOSSIM -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -x objective-c -I%t/InputsB %s  -verify

#ifdef ENABLE_IOSSIM
@import RequiresiOSSim;
#endif

// RUN: mkdir %t/InputsC
// RUN: echo "module RequiresLinuxEABIA {"  >> %t/InputsC/module.map
// RUN: echo "  requires linux, gnueabi"    >> %t/InputsC/module.map
// RUN: echo "}"                            >> %t/InputsC/module.map
// RUN: echo "module RequiresLinuxEABIB {"  >> %t/InputsC/module.map
// RUN: echo "  requires gnueabi"           >> %t/InputsC/module.map
// RUN: echo "}"                            >> %t/InputsC/module.map
// RUN: echo "module RequiresLinuxEABIC {"  >> %t/InputsC/module.map
// RUN: echo "  requires linux"             >> %t/InputsC/module.map
// RUN: echo "}"                            >> %t/InputsC/module.map
// RUN: %clang_cc1 -triple=armv8r-none-linux-gnueabi -DENABLE_LINUXEABI -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -x objective-c -I%t/InputsC %s -verify

#ifdef ENABLE_LINUXEABI
@import RequiresLinuxEABIA;
@import RequiresLinuxEABIB;
@import RequiresLinuxEABIC;
#endif

// RUN: mkdir %t/InputsD
// RUN: echo "module RequiresWinMSVCA {"  >> %t/InputsD/module.map
// RUN: echo "  requires windows"         >> %t/InputsD/module.map
// RUN: echo "}"                          >> %t/InputsD/module.map
// RUN: echo "module RequiresWinMSVCB {"  >> %t/InputsD/module.map
// RUN: echo "  requires windows, msvc"   >> %t/InputsD/module.map
// RUN: echo "}"                          >> %t/InputsD/module.map
// RUN: echo "module RequiresWinMSVCC {"  >> %t/InputsD/module.map
// RUN: echo "  requires msvc"            >> %t/InputsD/module.map
// RUN: echo "}"                          >> %t/InputsD/module.map
// RUN: %clang_cc1 -triple=thumbv7-unknown-windows-msvc -DENABLE_WINMSVC -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -x objective-c -I%t/InputsD %s  -verify

#ifdef ENABLE_WINMSVC
@import RequiresWinMSVCA;
@import RequiresWinMSVCB;
@import RequiresWinMSVCC;
#endif
