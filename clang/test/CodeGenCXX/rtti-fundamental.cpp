// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -emit-llvm -fvisibility hidden -o - | FileCheck %s -check-prefix=CHECK-HIDDEN

#include <typeinfo>

std::type_info foo() {
  return typeid(void);
}

namespace __cxxabiv1 {
  struct __fundamental_type_info {
    virtual ~__fundamental_type_info();
  };

  __fundamental_type_info::~__fundamental_type_info() { }
}

// void
// CHECK: @_ZTIv ={{.*}} constant
// CHECK-HIDDEN: @_ZTIv = hidden constant
// CHECK: @_ZTIPv ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPv = hidden constant
// CHECK: @_ZTIPKv ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKv = hidden constant

// std::nullptr_t
// CHECK: @_ZTIDn ={{.*}} constant
// CHECK-HIDDEN: @_ZTIDn = hidden constant
// CHECK: @_ZTIPDn ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPDn = hidden constant
// CHECK: @_ZTIPKDn ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKDn = hidden constant

// bool
// CHECK: @_ZTIb ={{.*}} constant
// CHECK-HIDDEN: @_ZTIb = hidden constant
// CHECK: @_ZTIPb ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPb = hidden constant
// CHECK: @_ZTIPKb ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKb = hidden constant

// wchar_t
// CHECK: @_ZTIw ={{.*}} constant
// CHECK-HIDDEN: @_ZTIw = hidden constant
// CHECK: @_ZTIPw ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPw = hidden constant
// CHECK: @_ZTIPKw ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKw = hidden constant

// char
// CHECK: @_ZTIc ={{.*}} constant
// CHECK-HIDDEN: @_ZTIc = hidden constant
// CHECK: @_ZTIPc ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPc = hidden constant
// CHECK: @_ZTIPKc ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKc = hidden constant

// unsigned char
// CHECK: @_ZTIh ={{.*}} constant
// CHECK-HIDDEN: @_ZTIh = hidden constant
// CHECK: @_ZTIPh ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPh = hidden constant
// CHECK: @_ZTIPKh ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKh = hidden constant

// signed char
// CHECK: @_ZTIa ={{.*}} constant
// CHECK-HIDDEN: @_ZTIa = hidden constant
// CHECK: @_ZTIPa ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPa = hidden constant
// CHECK: @_ZTIPKa ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKa = hidden constant

// short
// CHECK: @_ZTIs ={{.*}} constant
// CHECK-HIDDEN: @_ZTIs = hidden constant
// CHECK: @_ZTIPs ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPs = hidden constant
// CHECK: @_ZTIPKs ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKs = hidden constant

// unsigned short
// CHECK: @_ZTIt ={{.*}} constant
// CHECK-HIDDEN: @_ZTIt = hidden constant
// CHECK: @_ZTIPt ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPt = hidden constant
// CHECK: @_ZTIPKt ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKt = hidden constant

// int
// CHECK: @_ZTIi ={{.*}} constant
// CHECK-HIDDEN: @_ZTIi = hidden constant
// CHECK: @_ZTIPi ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPi = hidden constant
// CHECK: @_ZTIPKi ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKi = hidden constant

// unsigned int
// CHECK: @_ZTIj ={{.*}} constant
// CHECK-HIDDEN: @_ZTIj = hidden constant
// CHECK: @_ZTIPj ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPj = hidden constant
// CHECK: @_ZTIPKj ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKj = hidden constant

// long
// CHECK: @_ZTIl ={{.*}} constant
// CHECK-HIDDEN: @_ZTIl = hidden constant
// CHECK: @_ZTIPl ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPl = hidden constant
// CHECK: @_ZTIPKl ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKl = hidden constant

// unsigned long
// CHECK: @_ZTIm ={{.*}} constant
// CHECK-HIDDEN: @_ZTIm = hidden constant
// CHECK: @_ZTIPm ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPm = hidden constant
// CHECK: @_ZTIPKm ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKm = hidden constant

// long long
// CHECK: @_ZTIx ={{.*}} constant
// CHECK-HIDDEN: @_ZTIx = hidden constant
// CHECK: @_ZTIPx ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPx = hidden constant
// CHECK: @_ZTIPKx ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKx = hidden constant

// unsigned long long
// CHECK: @_ZTIy ={{.*}} constant
// CHECK-HIDDEN: @_ZTIy = hidden constant
// CHECK: @_ZTIPy ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPy = hidden constant
// CHECK: @_ZTIPKy ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKy = hidden constant

// __int128
// CHECK: @_ZTIn ={{.*}} constant
// CHECK-HIDDEN: @_ZTIn = hidden constant
// CHECK: @_ZTIPn ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPn = hidden constant
// CHECK: @_ZTIPKn ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKn = hidden constant

// unsigned __int128
// CHECK: @_ZTIo ={{.*}} constant
// CHECK-HIDDEN: @_ZTIo = hidden constant
// CHECK: @_ZTIPo ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPo = hidden constant
// CHECK: @_ZTIPKo ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKo = hidden constant

// half
// CHECK: @_ZTIDh ={{.*}} constant
// CHECK-HIDDEN: @_ZTIDh = hidden constant
// CHECK: @_ZTIPDh ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPDh = hidden constant
// CHECK: @_ZTIPKDh ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKDh = hidden constant

// float
// CHECK: @_ZTIf ={{.*}} constant
// CHECK-HIDDEN: @_ZTIf = hidden constant
// CHECK: @_ZTIPf ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPf = hidden constant
// CHECK: @_ZTIPKf ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKf = hidden constant

// double
// CHECK: @_ZTId ={{.*}} constant
// CHECK-HIDDEN: @_ZTId = hidden constant
// CHECK: @_ZTIPd ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPd = hidden constant
// CHECK: @_ZTIPKd ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKd = hidden constant

// long double
// CHECK: @_ZTIe ={{.*}} constant
// CHECK-HIDDEN: @_ZTIe = hidden constant
// CHECK: @_ZTIPe ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPe = hidden constant
// CHECK: @_ZTIPKe ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKe = hidden constant

// char16_t
// CHECK: @_ZTIDs ={{.*}} constant
// CHECK-HIDDEN: @_ZTIDs = hidden constant
// CHECK: @_ZTIPDs ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPDs = hidden constant
// CHECK: @_ZTIPKDs ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKDs = hidden constant

// char32_t
// CHECK: @_ZTIDi ={{.*}} constant
// CHECK-HIDDEN: @_ZTIDi = hidden constant
// CHECK: @_ZTIPDi ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPDi = hidden constant
// CHECK: @_ZTIPKDi ={{.*}} constant
// CHECK-HIDDEN: @_ZTIPKDi = hidden constant
