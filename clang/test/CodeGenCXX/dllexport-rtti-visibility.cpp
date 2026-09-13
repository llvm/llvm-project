/// dllexport expresses non-hidden intention and takes precedence over the
/// visibility implied by -fvisibility=hidden. Check that this holds for RTTI.

// RUN: %clang_cc1 -emit-llvm -triple x86_64-windows-gnu -fdeclspec -fvisibility=hidden -o - %s | FileCheck %s --check-prefixes=CHECK,GNU
// RUN: %clang_cc1 -emit-llvm -triple x86_64-windows-itanium -fdeclspec -fvisibility=hidden -o - %s | FileCheck %s --check-prefixes=CHECK,ITANIUM
// RUN: %clang_cc1 -emit-llvm -triple x86_64-scei-ps4 -fdeclspec -fvisibility=hidden -o - %s | FileCheck %s --check-prefix=PS
// RUN: %clang_cc1 -emit-llvm -triple x86_64-sie-ps5 -fdeclspec -fvisibility=hidden -o - %s | FileCheck %s --check-prefix=PS

// GNU-DAG: @_ZTI5plain = linkonce_odr hidden constant
// GNU-DAG: @_ZTS5plain = linkonce_odr hidden constant
// ITANIUM-DAG: @_ZTI5plain = hidden constant
// ITANIUM-DAG: @_ZTS5plain = hidden constant
// PS-DAG: @_ZTI5plain = {{(linkonce_odr )?}}hidden constant
// PS-DAG: @_ZTS5plain = {{(linkonce_odr )?}}hidden constant
struct plain {
  virtual ~plain();
};
plain::~plain() {}

/// RTTI is only dllexported for Windows Itanium; PS4/PS5 export the type info
/// but not the type name.
// GNU-DAG: @_ZTI8exported = linkonce_odr hidden constant
// GNU-DAG: @_ZTS8exported = linkonce_odr hidden constant
// ITANIUM-DAG: @_ZTI8exported = dso_local dllexport constant
// ITANIUM-DAG: @_ZTS8exported = dso_local dllexport constant
// PS-DAG: @_ZTI8exported = {{(dso_local )?}}dllexport constant
// PS-DAG: @_ZTS8exported = {{(linkonce_odr )?}}hidden constant
struct __declspec(dllexport) exported {
  virtual ~exported();
};
exported::~exported() {}

/// PS4/PS5 also export the type info when a non-inline method is dllexported.
// GNU-DAG: @_ZTI10partExport = linkonce_odr hidden constant
// GNU-DAG: @_ZTS10partExport = linkonce_odr hidden constant
// ITANIUM-DAG: @_ZTI10partExport = hidden constant
// ITANIUM-DAG: @_ZTS10partExport = hidden constant
// PS-DAG: @_ZTI10partExport = {{(dso_local )?}}dllexport constant
// PS-DAG: @_ZTS10partExport = {{(linkonce_odr )?}}hidden constant
struct partExport {
  virtual ~partExport();
  __declspec(dllexport) void f();
};
partExport::~partExport() {}

/// Defining __cxxabiv1::__fundamental_type_info makes Clang implicitly emit the
/// RTTI descriptors for the fundamental types, with the class' storage class.
// GNU-DAG: @_ZTIN10__cxxabiv123__fundamental_type_infoE = linkonce_odr hidden constant
// GNU-DAG: @_ZTSN10__cxxabiv123__fundamental_type_infoE = linkonce_odr hidden constant
// ITANIUM-DAG: @_ZTIN10__cxxabiv123__fundamental_type_infoE = dso_local dllexport constant
// ITANIUM-DAG: @_ZTSN10__cxxabiv123__fundamental_type_infoE = dso_local dllexport constant
// PS-DAG: @_ZTIN10__cxxabiv123__fundamental_type_infoE = {{(dso_local )?}}dllexport constant
// PS-DAG: @_ZTSN10__cxxabiv123__fundamental_type_infoE = {{(linkonce_odr )?}}hidden constant

// PS-DAG: @_ZTIv = {{(dso_local )?}}dllexport constant
// PS-DAG: @_ZTSv = {{(dso_local )?}}dllexport constant

// CHECK-DAG: @_ZTIv = dso_local dllexport constant
// CHECK-DAG: @_ZTSv = dso_local dllexport constant
// CHECK-DAG: @_ZTIPv = dso_local dllexport constant
// CHECK-DAG: @_ZTSPv = dso_local dllexport constant
// CHECK-DAG: @_ZTIPKv = dso_local dllexport constant
// CHECK-DAG: @_ZTSPKv = dso_local dllexport constant
// CHECK-DAG: @_ZTIi = dso_local dllexport constant
// CHECK-DAG: @_ZTSi = dso_local dllexport constant
// CHECK-DAG: @_ZTIPi = dso_local dllexport constant
// CHECK-DAG: @_ZTSPi = dso_local dllexport constant
// CHECK-DAG: @_ZTIPKi = dso_local dllexport constant
// CHECK-DAG: @_ZTSPKi = dso_local dllexport constant
namespace __cxxabiv1 {
struct __declspec(dllexport) __fundamental_type_info {
  virtual ~__fundamental_type_info();
};
__fundamental_type_info::~__fundamental_type_info() {}
} // namespace __cxxabiv1
