// RUN: %clang_cc1 -triple powerpc64-ibm-aix %s -internal-isystem %S -mdefault-visibility-export-mapping=none -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=CHECK,UNSPECIFIED-DEF,EXPLICIT-DEF,FUND-DEF %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix %s -internal-isystem %S -mdefault-visibility-export-mapping=explicit -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=CHECK,UNSPECIFIED-DEF,EXPLICIT-EXP,FUND-DEF %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix %s -internal-isystem %S -mdefault-visibility-export-mapping=explicit -DFUNDAMENTAL_IS_EXPLICIT -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=CHECK,UNSPECIFIED-DEF,EXPLICIT-EXP,FUND-EXP %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix %s -internal-isystem %S -mdefault-visibility-export-mapping=all -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=CHECK,UNSPECIFIED-EXP,EXPLICIT-EXP,FUND-EXP %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix %s -internal-isystem %S -mdefault-visibility-export-mapping=all -fvisibility=hidden -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=CHECK,UNSPECIFIED-HID,EXPLICIT-EXP,FUND-HID %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix %s -internal-isystem %S -mdefault-visibility-export-mapping=all -DFUNDAMENTAL_IS_EXPLICIT -fvisibility=hidden -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=CHECK,UNSPECIFIED-HID,EXPLICIT-EXP,FUND-EXP %s

#include <typeinfo>

// C is an incomplete class type, so any direct or indirect pointer types should have 
// internal linkage, as should the type info for C itself.
struct C;
// CHECK: @_ZTIP1C = internal constant
// CHECK: @_ZTSP1C = internal constant
// CHECK: @_ZTI1C = internal constant
// CHECK: @_ZTS1C = internal constant
// CHECK: @_ZTIPP1C = internal constant
// CHECK: @_ZTSPP1C = internal constant

struct __attribute__((type_visibility("default"))) D;
// CHECK: @_ZTIP1D = internal constant
// CHECK: @_ZTSP1D = internal constant
// CHECK: @_ZTI1D = internal constant
// CHECK: @_ZTS1D = internal constant
// CHECK: @_ZTIPP1D = internal constant
// CHECK: @_ZTSPP1D = internal constant

void __attribute__((visibility("default"))) tfunc() {
  (void)typeid(C *);
  (void)typeid(C **);
  (void)typeid(D *);
  (void)typeid(D **);
}

// unspecified visibility RTTI & vtable
struct s {
  virtual void foo();
};
void s::foo() {}
// UNSPECIFIED-DEF: @_ZTV1s = unnamed_addr constant
// UNSPECIFIED-HID: @_ZTV1s = hidden unnamed_addr constant
// UNSPECIFIED-EXP: @_ZTV1s = dllexport unnamed_addr constant
// UNSPECIFIED-DEF: @_ZTI1s = constant
// UNSPECIFIED-HID: @_ZTI1s = hidden constant
// UNSPECIFIED-EXP: @_ZTI1s = dllexport constant
// UNSPECIFIED-DEF: @_ZTS1s = constant
// UNSPECIFIED-HID: @_ZTS1s = hidden constant
// UNSPECIFIED-EXP: @_ZTS1s = dllexport constant

// explicit default visibility RTTI & vtable
struct __attribute__((type_visibility("default"))) t {
  virtual void foo();
};
void t::foo() {}
// EXPLICIT-DEF: @_ZTV1t = unnamed_addr constant
// EXPLICIT-HID: @_ZTV1t = hidden unnamed_addr constant
// EXPLICIT-EXP: @_ZTV1t = dllexport unnamed_addr constant
// EXPLICIT-DEF: @_ZTI1t = constant
// EXPLICIT-HID: @_ZTI1t = hidden constant
// EXPLICIT-EXP: @_ZTI1t = dllexport constant
// EXPLICIT-DEF: @_ZTS1t = constant
// EXPLICIT-HID: @_ZTS1t = hidden constant
// EXPLICIT-EXP: @_ZTS1t = dllexport constant

#ifdef FUNDAMENTAL_IS_EXPLICIT
#define TYPE_VIS __attribute__((type_visibility("default")))
#else
#define TYPE_VIS
#endif

// Invoke the compiler magic to emit RTTI for fundamental types.
namespace __cxxabiv1 {
class TYPE_VIS __fundamental_type_info {
  __attribute__((visibility("hidden"))) virtual ~__fundamental_type_info();
};

__fundamental_type_info::~__fundamental_type_info() {}

} // namespace __cxxabiv1

// __cxxabiv1::__fundamental_type_info
// FUND-DEF: @_ZTVN10__cxxabiv123__fundamental_type_infoE = unnamed_addr constant
// FUND-DEF: @_ZTIN10__cxxabiv123__fundamental_type_infoE = constant
// FUND-DEF: @_ZTSN10__cxxabiv123__fundamental_type_infoE = constant
// FUND-HID: @_ZTVN10__cxxabiv123__fundamental_type_infoE = hidden unnamed_addr constant
// FUND-HID: @_ZTIN10__cxxabiv123__fundamental_type_infoE = hidden constant
// FUND-HID: @_ZTSN10__cxxabiv123__fundamental_type_infoE = hidden constant
// FUND-EXP: @_ZTVN10__cxxabiv123__fundamental_type_infoE = dllexport unnamed_addr constant
// FUND-EXP: @_ZTIN10__cxxabiv123__fundamental_type_infoE = dllexport constant
// FUND-EXP: @_ZTSN10__cxxabiv123__fundamental_type_infoE = dllexport constant

// void
// FUND-DEF: @_ZTIv = constant
// FUND-DEF: @_ZTSv = constant
// FUND-DEF: @_ZTIPv = constant
// FUND-DEF: @_ZTSPv = constant
// FUND-DEF: @_ZTIPKv = constant
// FUND-DEF: @_ZTSPKv = constant
// FUND-HID: @_ZTIv = hidden constant
// FUND-HID: @_ZTSv = hidden constant
// FUND-HID: @_ZTIPv = hidden constant
// FUND-HID: @_ZTSPv = hidden constant
// FUND-HID: @_ZTIPKv = hidden constant
// FUND-HID: @_ZTSPKv = hidden constant
// FUND-EXP: @_ZTIv = dllexport constant
// FUND-EXP: @_ZTSv = dllexport constant
// FUND-EXP: @_ZTIPv = dllexport constant
// FUND-EXP: @_ZTSPv = dllexport constant
// FUND-EXP: @_ZTIPKv = dllexport constant
// FUND-EXP: @_ZTSPKv = dllexport constant

// std::nullptr_t
// FUND-DEF: @_ZTIDn = constant
// FUND-DEF: @_ZTSDn = constant
// FUND-DEF: @_ZTIPDn = constant
// FUND-DEF: @_ZTSPDn = constant
// FUND-DEF: @_ZTIPKDn = constant
// FUND-DEF: @_ZTSPKDn = constant
// FUND-HID: @_ZTIDn = hidden constant
// FUND-HID: @_ZTSDn = hidden constant
// FUND-HID: @_ZTIPDn = hidden constant
// FUND-HID: @_ZTSPDn = hidden constant
// FUND-HID: @_ZTIPKDn = hidden constant
// FUND-HID: @_ZTSPKDn = hidden constant
// FUND-EXP: @_ZTIDn = dllexport constant
// FUND-EXP: @_ZTSDn = dllexport constant
// FUND-EXP: @_ZTIPDn = dllexport constant
// FUND-EXP: @_ZTSPDn = dllexport constant
// FUND-EXP: @_ZTIPKDn = dllexport constant
// FUND-EXP: @_ZTSPKDn = dllexport constant

// bool
// FUND-DEF: @_ZTIb = constant
// FUND-DEF: @_ZTSb = constant
// FUND-DEF: @_ZTIPb = constant
// FUND-DEF: @_ZTSPb = constant
// FUND-DEF: @_ZTIPKb = constant
// FUND-DEF: @_ZTSPKb = constant
// FUND-HID: @_ZTIb = hidden constant
// FUND-HID: @_ZTSb = hidden constant
// FUND-HID: @_ZTIPb = hidden constant
// FUND-HID: @_ZTSPb = hidden constant
// FUND-HID: @_ZTIPKb = hidden constant
// FUND-HID: @_ZTSPKb = hidden constant
// FUND-EXP: @_ZTIb = dllexport constant
// FUND-EXP: @_ZTSb = dllexport constant
// FUND-EXP: @_ZTIPb = dllexport constant
// FUND-EXP: @_ZTSPb = dllexport constant
// FUND-EXP: @_ZTIPKb = dllexport constant
// FUND-EXP: @_ZTSPKb = dllexport constant

// wchar_t
// FUND-DEF: @_ZTIw = constant
// FUND-DEF: @_ZTSw = constant
// FUND-DEF: @_ZTIPw = constant
// FUND-DEF: @_ZTSPw = constant
// FUND-DEF: @_ZTIPKw = constant
// FUND-DEF: @_ZTSPKw = constant
// FUND-HID: @_ZTIw = hidden constant
// FUND-HID: @_ZTSw = hidden constant
// FUND-HID: @_ZTIPw = hidden constant
// FUND-HID: @_ZTSPw = hidden constant
// FUND-HID: @_ZTIPKw = hidden constant
// FUND-HID: @_ZTSPKw = hidden constant
// FUND-EXP: @_ZTIw = dllexport constant
// FUND-EXP: @_ZTSw = dllexport constant
// FUND-EXP: @_ZTIPw = dllexport constant
// FUND-EXP: @_ZTSPw = dllexport constant
// FUND-EXP: @_ZTIPKw = dllexport constant
// FUND-EXP: @_ZTSPKw = dllexport constant

// char
// FUND-DEF: @_ZTIc = constant
// FUND-DEF: @_ZTSc = constant
// FUND-DEF: @_ZTIPc = constant
// FUND-DEF: @_ZTSPc = constant
// FUND-DEF: @_ZTIPKc = constant
// FUND-DEF: @_ZTSPKc = constant
// FUND-HID: @_ZTIc = hidden constant
// FUND-HID: @_ZTSc = hidden constant
// FUND-HID: @_ZTIPc = hidden constant
// FUND-HID: @_ZTSPc = hidden constant
// FUND-HID: @_ZTIPKc = hidden constant
// FUND-HID: @_ZTSPKc = hidden constant
// FUND-EXP: @_ZTIc = dllexport constant
// FUND-EXP: @_ZTSc = dllexport constant
// FUND-EXP: @_ZTIPc = dllexport constant
// FUND-EXP: @_ZTSPc = dllexport constant
// FUND-EXP: @_ZTIPKc = dllexport constant
// FUND-EXP: @_ZTSPKc = dllexport constant

// unsigned char
// FUND-DEF: @_ZTIh = constant
// FUND-DEF: @_ZTSh = constant
// FUND-DEF: @_ZTIPh = constant
// FUND-DEF: @_ZTSPh = constant
// FUND-DEF: @_ZTIPKh = constant
// FUND-DEF: @_ZTSPKh = constant
// FUND-HID: @_ZTIh = hidden constant
// FUND-HID: @_ZTSh = hidden constant
// FUND-HID: @_ZTIPh = hidden constant
// FUND-HID: @_ZTSPh = hidden constant
// FUND-HID: @_ZTIPKh = hidden constant
// FUND-HID: @_ZTSPKh = hidden constant
// FUND-EXP: @_ZTIh = dllexport constant
// FUND-EXP: @_ZTSh = dllexport constant
// FUND-EXP: @_ZTIPh = dllexport constant
// FUND-EXP: @_ZTSPh = dllexport constant
// FUND-EXP: @_ZTIPKh = dllexport constant
// FUND-EXP: @_ZTSPKh = dllexport constant

// signed char
// FUND-DEF: @_ZTIa = constant
// FUND-DEF: @_ZTSa = constant
// FUND-DEF: @_ZTIPa = constant
// FUND-DEF: @_ZTSPa = constant
// FUND-DEF: @_ZTIPKa = constant
// FUND-DEF: @_ZTSPKa = constant
// FUND-HID: @_ZTIa = hidden constant
// FUND-HID: @_ZTSa = hidden constant
// FUND-HID: @_ZTIPa = hidden constant
// FUND-HID: @_ZTSPa = hidden constant
// FUND-HID: @_ZTIPKa = hidden constant
// FUND-HID: @_ZTSPKa = hidden constant
// FUND-EXP: @_ZTIa = dllexport constant
// FUND-EXP: @_ZTSa = dllexport constant
// FUND-EXP: @_ZTIPa = dllexport constant
// FUND-EXP: @_ZTSPa = dllexport constant
// FUND-EXP: @_ZTIPKa = dllexport constant
// FUND-EXP: @_ZTSPKa = dllexport constant

// short
// FUND-DEF: @_ZTIs = constant
// FUND-DEF: @_ZTSs = constant
// FUND-DEF: @_ZTIPs = constant
// FUND-DEF: @_ZTSPs = constant
// FUND-DEF: @_ZTIPKs = constant
// FUND-DEF: @_ZTSPKs = constant
// FUND-HID: @_ZTIs = hidden constant
// FUND-HID: @_ZTSs = hidden constant
// FUND-HID: @_ZTIPs = hidden constant
// FUND-HID: @_ZTSPs = hidden constant
// FUND-HID: @_ZTIPKs = hidden constant
// FUND-HID: @_ZTSPKs = hidden constant
// FUND-EXP: @_ZTIs = dllexport constant
// FUND-EXP: @_ZTSs = dllexport constant
// FUND-EXP: @_ZTIPs = dllexport constant
// FUND-EXP: @_ZTSPs = dllexport constant
// FUND-EXP: @_ZTIPKs = dllexport constant
// FUND-EXP: @_ZTSPKs = dllexport constant

// unsigned short
// FUND-DEF: @_ZTIt = constant
// FUND-DEF: @_ZTSt = constant
// FUND-DEF: @_ZTIPt = constant
// FUND-DEF: @_ZTSPt = constant
// FUND-DEF: @_ZTIPKt = constant
// FUND-DEF: @_ZTSPKt = constant
// FUND-HID: @_ZTIt = hidden constant
// FUND-HID: @_ZTSt = hidden constant
// FUND-HID: @_ZTIPt = hidden constant
// FUND-HID: @_ZTSPt = hidden constant
// FUND-HID: @_ZTIPKt = hidden constant
// FUND-HID: @_ZTSPKt = hidden constant
// FUND-EXP: @_ZTIt = dllexport constant
// FUND-EXP: @_ZTSt = dllexport constant
// FUND-EXP: @_ZTIPt = dllexport constant
// FUND-EXP: @_ZTSPt = dllexport constant
// FUND-EXP: @_ZTIPKt = dllexport constant
// FUND-EXP: @_ZTSPKt = dllexport constant

// int
// FUND-DEF: @_ZTIi = constant
// FUND-DEF: @_ZTSi = constant
// FUND-DEF: @_ZTIPi = constant
// FUND-DEF: @_ZTSPi = constant
// FUND-DEF: @_ZTIPKi = constant
// FUND-DEF: @_ZTSPKi = constant
// FUND-HID: @_ZTIi = hidden constant
// FUND-HID: @_ZTSi = hidden constant
// FUND-HID: @_ZTIPi = hidden constant
// FUND-HID: @_ZTSPi = hidden constant
// FUND-HID: @_ZTIPKi = hidden constant
// FUND-HID: @_ZTSPKi = hidden constant
// FUND-EXP: @_ZTIi = dllexport constant
// FUND-EXP: @_ZTSi = dllexport constant
// FUND-EXP: @_ZTIPi = dllexport constant
// FUND-EXP: @_ZTSPi = dllexport constant
// FUND-EXP: @_ZTIPKi = dllexport constant
// FUND-EXP: @_ZTSPKi = dllexport constant

// unsigned int
// FUND-DEF: @_ZTIj = constant
// FUND-DEF: @_ZTSj = constant
// FUND-DEF: @_ZTIPj = constant
// FUND-DEF: @_ZTSPj = constant
// FUND-DEF: @_ZTIPKj = constant
// FUND-DEF: @_ZTSPKj = constant
// FUND-HID: @_ZTIj = hidden constant
// FUND-HID: @_ZTSj = hidden constant
// FUND-HID: @_ZTIPj = hidden constant
// FUND-HID: @_ZTSPj = hidden constant
// FUND-HID: @_ZTIPKj = hidden constant
// FUND-HID: @_ZTSPKj = hidden constant
// FUND-EXP: @_ZTIj = dllexport constant
// FUND-EXP: @_ZTSj = dllexport constant
// FUND-EXP: @_ZTIPj = dllexport constant
// FUND-EXP: @_ZTSPj = dllexport constant
// FUND-EXP: @_ZTIPKj = dllexport constant
// FUND-EXP: @_ZTSPKj = dllexport constant

// long
// FUND-DEF: @_ZTIl = constant
// FUND-DEF: @_ZTSl = constant
// FUND-DEF: @_ZTIPl = constant
// FUND-DEF: @_ZTSPl = constant
// FUND-DEF: @_ZTIPKl = constant
// FUND-DEF: @_ZTSPKl = constant
// FUND-HID: @_ZTIl = hidden constant
// FUND-HID: @_ZTSl = hidden constant
// FUND-HID: @_ZTIPl = hidden constant
// FUND-HID: @_ZTSPl = hidden constant
// FUND-HID: @_ZTIPKl = hidden constant
// FUND-HID: @_ZTSPKl = hidden constant
// FUND-EXP: @_ZTIl = dllexport constant
// FUND-EXP: @_ZTSl = dllexport constant
// FUND-EXP: @_ZTIPl = dllexport constant
// FUND-EXP: @_ZTSPl = dllexport constant
// FUND-EXP: @_ZTIPKl = dllexport constant
// FUND-EXP: @_ZTSPKl = dllexport constant

// unsigned long
// FUND-DEF: @_ZTIm = constant
// FUND-DEF: @_ZTSm = constant
// FUND-DEF: @_ZTIPm = constant
// FUND-DEF: @_ZTSPm = constant
// FUND-DEF: @_ZTIPKm = constant
// FUND-DEF: @_ZTSPKm = constant
// FUND-HID: @_ZTIm = hidden constant
// FUND-HID: @_ZTSm = hidden constant
// FUND-HID: @_ZTIPm = hidden constant
// FUND-HID: @_ZTSPm = hidden constant
// FUND-HID: @_ZTIPKm = hidden constant
// FUND-HID: @_ZTSPKm = hidden constant
// FUND-EXP: @_ZTIm = dllexport constant
// FUND-EXP: @_ZTSm = dllexport constant
// FUND-EXP: @_ZTIPm = dllexport constant
// FUND-EXP: @_ZTSPm = dllexport constant
// FUND-EXP: @_ZTIPKm = dllexport constant
// FUND-EXP: @_ZTSPKm = dllexport constant

// long long
// FUND-DEF: @_ZTIx = constant
// FUND-DEF: @_ZTSx = constant
// FUND-DEF: @_ZTIPx = constant
// FUND-DEF: @_ZTSPx = constant
// FUND-DEF: @_ZTIPKx = constant
// FUND-DEF: @_ZTSPKx = constant
// FUND-HID: @_ZTIx = hidden constant
// FUND-HID: @_ZTSx = hidden constant
// FUND-HID: @_ZTIPx = hidden constant
// FUND-HID: @_ZTSPx = hidden constant
// FUND-HID: @_ZTIPKx = hidden constant
// FUND-HID: @_ZTSPKx = hidden constant
// FUND-EXP: @_ZTIx = dllexport constant
// FUND-EXP: @_ZTSx = dllexport constant
// FUND-EXP: @_ZTIPx = dllexport constant
// FUND-EXP: @_ZTSPx = dllexport constant
// FUND-EXP: @_ZTIPKx = dllexport constant
// FUND-EXP: @_ZTSPKx = dllexport constant

// unsigned long long
// FUND-DEF: @_ZTIy = constant
// FUND-DEF: @_ZTSy = constant
// FUND-DEF: @_ZTIPy = constant
// FUND-DEF: @_ZTSPy = constant
// FUND-DEF: @_ZTIPKy = constant
// FUND-DEF: @_ZTSPKy = constant
// FUND-HID: @_ZTIy = hidden constant
// FUND-HID: @_ZTSy = hidden constant
// FUND-HID: @_ZTIPy = hidden constant
// FUND-HID: @_ZTSPy = hidden constant
// FUND-HID: @_ZTIPKy = hidden constant
// FUND-HID: @_ZTSPKy = hidden constant
// FUND-EXP: @_ZTIy = dllexport constant
// FUND-EXP: @_ZTSy = dllexport constant
// FUND-EXP: @_ZTIPy = dllexport constant
// FUND-EXP: @_ZTSPy = dllexport constant
// FUND-EXP: @_ZTIPKy = dllexport constant
// FUND-EXP: @_ZTSPKy = dllexport constant

// __int128
// FUND-DEF: @_ZTIn = constant
// FUND-DEF: @_ZTSn = constant
// FUND-DEF: @_ZTIPn = constant
// FUND-DEF: @_ZTSPn = constant
// FUND-DEF: @_ZTIPKn = constant
// FUND-DEF: @_ZTSPKn = constant
// FUND-HID: @_ZTIn = hidden constant
// FUND-HID: @_ZTSn = hidden constant
// FUND-HID: @_ZTIPn = hidden constant
// FUND-HID: @_ZTSPn = hidden constant
// FUND-HID: @_ZTIPKn = hidden constant
// FUND-HID: @_ZTSPKn = hidden constant
// FUND-EXP: @_ZTIn = dllexport constant
// FUND-EXP: @_ZTSn = dllexport constant
// FUND-EXP: @_ZTIPn = dllexport constant
// FUND-EXP: @_ZTSPn = dllexport constant
// FUND-EXP: @_ZTIPKn = dllexport constant
// FUND-EXP: @_ZTSPKn = dllexport constant

// unsigned __int128
// FUND-DEF: @_ZTIo = constant
// FUND-DEF: @_ZTSo = constant
// FUND-DEF: @_ZTIPo = constant
// FUND-DEF: @_ZTSPo = constant
// FUND-DEF: @_ZTIPKo = constant
// FUND-DEF: @_ZTSPKo = constant
// FUND-HID: @_ZTIo = hidden constant
// FUND-HID: @_ZTSo = hidden constant
// FUND-HID: @_ZTIPo = hidden constant
// FUND-HID: @_ZTSPo = hidden constant
// FUND-HID: @_ZTIPKo = hidden constant
// FUND-HID: @_ZTSPKo = hidden constant
// FUND-EXP: @_ZTIo = dllexport constant
// FUND-EXP: @_ZTSo = dllexport constant
// FUND-EXP: @_ZTIPo = dllexport constant
// FUND-EXP: @_ZTSPo = dllexport constant
// FUND-EXP: @_ZTIPKo = dllexport constant
// FUND-EXP: @_ZTSPKo = dllexport constant

// half
// FUND-DEF: @_ZTIDh = constant
// FUND-DEF: @_ZTSDh = constant
// FUND-DEF: @_ZTIPDh = constant
// FUND-DEF: @_ZTSPDh = constant
// FUND-DEF: @_ZTIPKDh = constant
// FUND-DEF: @_ZTSPKDh = constant
// FUND-HID: @_ZTIDh = hidden constant
// FUND-HID: @_ZTSDh = hidden constant
// FUND-HID: @_ZTIPDh = hidden constant
// FUND-HID: @_ZTSPDh = hidden constant
// FUND-HID: @_ZTIPKDh = hidden constant
// FUND-HID: @_ZTSPKDh = hidden constant
// FUND-EXP: @_ZTIDh = dllexport constant
// FUND-EXP: @_ZTSDh = dllexport constant
// FUND-EXP: @_ZTIPDh = dllexport constant
// FUND-EXP: @_ZTSPDh = dllexport constant
// FUND-EXP: @_ZTIPKDh = dllexport constant
// FUND-EXP: @_ZTSPKDh = dllexport constant

// float
// FUND-DEF: @_ZTIf = constant
// FUND-DEF: @_ZTSf = constant
// FUND-DEF: @_ZTIPf = constant
// FUND-DEF: @_ZTSPf = constant
// FUND-DEF: @_ZTIPKf = constant
// FUND-DEF: @_ZTSPKf = constant
// FUND-HID: @_ZTIf = hidden constant
// FUND-HID: @_ZTSf = hidden constant
// FUND-HID: @_ZTIPf = hidden constant
// FUND-HID: @_ZTSPf = hidden constant
// FUND-HID: @_ZTIPKf = hidden constant
// FUND-HID: @_ZTSPKf = hidden constant
// FUND-EXP: @_ZTIf = dllexport constant
// FUND-EXP: @_ZTSf = dllexport constant
// FUND-EXP: @_ZTIPf = dllexport constant
// FUND-EXP: @_ZTSPf = dllexport constant
// FUND-EXP: @_ZTIPKf = dllexport constant
// FUND-EXP: @_ZTSPKf = dllexport constant

// double
// FUND-DEF: @_ZTId = constant
// FUND-DEF: @_ZTSd = constant
// FUND-DEF: @_ZTIPd = constant
// FUND-DEF: @_ZTSPd = constant
// FUND-DEF: @_ZTIPKd = constant
// FUND-DEF: @_ZTSPKd = constant
// FUND-HID: @_ZTId = hidden constant
// FUND-HID: @_ZTSd = hidden constant
// FUND-HID: @_ZTIPd = hidden constant
// FUND-HID: @_ZTSPd = hidden constant
// FUND-HID: @_ZTIPKd = hidden constant
// FUND-HID: @_ZTSPKd = hidden constant
// FUND-EXP: @_ZTId = dllexport constant
// FUND-EXP: @_ZTSd = dllexport constant
// FUND-EXP: @_ZTIPd = dllexport constant
// FUND-EXP: @_ZTSPd = dllexport constant
// FUND-EXP: @_ZTIPKd = dllexport constant
// FUND-EXP: @_ZTSPKd = dllexport constant

// long double
// FUND-DEF: @_ZTIe = constant
// FUND-DEF: @_ZTSe = constant
// FUND-DEF: @_ZTIPe = constant
// FUND-DEF: @_ZTSPe = constant
// FUND-DEF: @_ZTIPKe = constant
// FUND-DEF: @_ZTSPKe = constant
// FUND-HID: @_ZTIe = hidden constant
// FUND-HID: @_ZTSe = hidden constant
// FUND-HID: @_ZTIPe = hidden constant
// FUND-HID: @_ZTSPe = hidden constant
// FUND-HID: @_ZTIPKe = hidden constant
// FUND-HID: @_ZTSPKe = hidden constant
// FUND-EXP: @_ZTIe = dllexport constant
// FUND-EXP: @_ZTSe = dllexport constant
// FUND-EXP: @_ZTIPe = dllexport constant
// FUND-EXP: @_ZTSPe = dllexport constant
// FUND-EXP: @_ZTIPKe = dllexport constant
// FUND-EXP: @_ZTSPKe = dllexport constant

// __ieee128
// FUND-DEF: @_ZTIu9__ieee128 = constant
// FUND-DEF: @_ZTSu9__ieee128 = constant
// FUND-DEF: @_ZTIPu9__ieee128 = constant
// FUND-DEF: @_ZTSPu9__ieee128 = constant
// FUND-DEF: @_ZTIPKu9__ieee128 = constant
// FUND-DEF: @_ZTSPKu9__ieee128 = constant
// FUND-HID: @_ZTIu9__ieee128 = hidden constant
// FUND-HID: @_ZTSu9__ieee128 = hidden constant
// FUND-HID: @_ZTIPu9__ieee128 = hidden constant
// FUND-HID: @_ZTSPu9__ieee128 = hidden constant
// FUND-HID: @_ZTIPKu9__ieee128 = hidden constant
// FUND-HID: @_ZTSPKu9__ieee128 = hidden constant
// FUND-EXP: @_ZTIu9__ieee128 = dllexport constant
// FUND-EXP: @_ZTSu9__ieee128 = dllexport constant
// FUND-EXP: @_ZTIPu9__ieee128 = dllexport constant
// FUND-EXP: @_ZTSPu9__ieee128 = dllexport constant
// FUND-EXP: @_ZTIPKu9__ieee128 = dllexport constant
// FUND-EXP: @_ZTSPKu9__ieee128 = dllexport constant

// char8_t
// FUND-DEF: @_ZTIDu = constant
// FUND-DEF: @_ZTSDu = constant
// FUND-DEF: @_ZTIPDu = constant
// FUND-DEF: @_ZTSPDu = constant
// FUND-DEF: @_ZTIPKDu = constant
// FUND-DEF: @_ZTSPKDu = constant
// FUND-HID: @_ZTIDu = hidden constant
// FUND-HID: @_ZTSDu = hidden constant
// FUND-HID: @_ZTIPDu = hidden constant
// FUND-HID: @_ZTSPDu = hidden constant
// FUND-HID: @_ZTIPKDu = hidden constant
// FUND-HID: @_ZTSPKDu = hidden constant
// FUND-EXP: @_ZTIDu = dllexport constant
// FUND-EXP: @_ZTSDu = dllexport constant
// FUND-EXP: @_ZTIPDu = dllexport constant
// FUND-EXP: @_ZTSPDu = dllexport constant
// FUND-EXP: @_ZTIPKDu = dllexport constant
// FUND-EXP: @_ZTSPKDu = dllexport constant

// char16_t
// FUND-DEF: @_ZTIDs = constant
// FUND-DEF: @_ZTSDs = constant
// FUND-DEF: @_ZTIPDs = constant
// FUND-DEF: @_ZTSPDs = constant
// FUND-DEF: @_ZTIPKDs = constant
// FUND-DEF: @_ZTSPKDs = constant
// FUND-HID: @_ZTIDs = hidden constant
// FUND-HID: @_ZTSDs = hidden constant
// FUND-HID: @_ZTIPDs = hidden constant
// FUND-HID: @_ZTSPDs = hidden constant
// FUND-HID: @_ZTIPKDs = hidden constant
// FUND-HID: @_ZTSPKDs = hidden constant
// FUND-EXP: @_ZTIDs = dllexport constant
// FUND-EXP: @_ZTSDs = dllexport constant
// FUND-EXP: @_ZTIPDs = dllexport constant
// FUND-EXP: @_ZTSPDs = dllexport constant
// FUND-EXP: @_ZTIPKDs = dllexport constant
// FUND-EXP: @_ZTSPKDs = dllexport constant

// char32_t
// FUND-DEF: @_ZTIDi = constant
// FUND-DEF: @_ZTSDi = constant
// FUND-DEF: @_ZTIPDi = constant
// FUND-DEF: @_ZTSPDi = constant
// FUND-DEF: @_ZTIPKDi = constant
// FUND-DEF: @_ZTSPKDi = constant
// FUND-HID: @_ZTIDi = hidden constant
// FUND-HID: @_ZTSDi = hidden constant
// FUND-HID: @_ZTIPDi = hidden constant
// FUND-HID: @_ZTSPDi = hidden constant
// FUND-HID: @_ZTIPKDi = hidden constant
// FUND-HID: @_ZTSPKDi = hidden constant
// FUND-EXP: @_ZTIDi = dllexport constant
// FUND-EXP: @_ZTSDi = dllexport constant
// FUND-EXP: @_ZTIPDi = dllexport constant
// FUND-EXP: @_ZTSPDi = dllexport constant
// FUND-EXP: @_ZTIPKDi = dllexport constant
// FUND-EXP: @_ZTSPKDi = dllexport constant
