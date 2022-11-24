// Initialisation Guard Variables should take their DLL storage class from
// the guarded variable. Otherwise, there will be a link error if the compiler
// inlines a reference to the guard variable into another module but that
// guard variable is not exported from the defining module.

// RUN: %clang_cc1 -emit-llvm -triple i686-windows-itanium -fdeclspec %s -O1 -disable-llvm-passes -o - -DAPI= | FileCheck %s --check-prefixes=NONE
// RUN: %clang_cc1 -emit-llvm -triple i686-windows-itanium -fdeclspec %s -O1 -disable-llvm-passes -o - -DAPI="__declspec(dllexport)" | FileCheck %s --check-prefixes=EXPORT
// RUN: %clang_cc1 -emit-llvm -triple i686-windows-itanium -fdeclspec %s -O1 -disable-llvm-passes -o - -DAPI="__declspec(dllimport)" | FileCheck %s --check-prefixes=IMPORT

// RUN: %clang_cc1 -emit-llvm -triple x86_64-scei-ps4 -fdeclspec %s -O1 -disable-llvm-passes -o - -DAPI= | FileCheck %s --check-prefixes=NONE
// RUN: %clang_cc1 -emit-llvm -triple x86_64-scei-ps4 -fdeclspec %s -O1 -disable-llvm-passes -o - -DAPI="__declspec(dllexport)" | FileCheck %s --check-prefixes=EXPORT
// RUN: %clang_cc1 -emit-llvm -triple x86_64-scei-ps4 -fdeclspec %s -O1 -disable-llvm-passes -o - -DAPI="__declspec(dllimport)" | FileCheck %s --check-prefixes=IMPORT

// RUN: %clang_cc1 -emit-llvm -triple x86_64-scei-ps5 -fdeclspec %s -O1 -disable-llvm-passes -o - -DAPI= | FileCheck %s --check-prefixes=NONE
// RUN: %clang_cc1 -emit-llvm -triple x86_64-scei-ps5 -fdeclspec %s -O1 -disable-llvm-passes -o - -DAPI="__declspec(dllexport)" | FileCheck %s --check-prefixes=EXPORT
// RUN: %clang_cc1 -emit-llvm -triple x86_64-scei-ps5 -fdeclspec %s -O1 -disable-llvm-passes -o - -DAPI="__declspec(dllimport)" | FileCheck %s --check-prefixes=IMPORT

//NONE: @_ZZN3foo3GetEvE9Singleton = linkonce_odr {{(dso_local )?}}global
//NONE: @_ZGVZN3foo3GetEvE9Singleton = linkonce_odr {{(dso_local )?}}global

//EXPORT: @_ZZN3foo3GetEvE9Singleton = weak_odr {{(dso_local )?}}dllexport global
//EXPORT: @_ZGVZN3foo3GetEvE9Singleton = weak_odr {{(dso_local )?}}dllexport global

//IMPORT: @_ZZN3foo3GetEvE9Singleton = available_externally dllimport global
//IMPORT: @_ZGVZN3foo3GetEvE9Singleton = available_externally dllimport global

struct API foo {
  foo() {}
  static void Get() { static foo Singleton; }
};

void f() { foo::Get(); }
