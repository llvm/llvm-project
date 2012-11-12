// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

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
// CHECK: @_ZTIv = unnamed_addr constant
// CHECK: @_ZTIPv = unnamed_addr constant
// CHECK: @_ZTIPKv = unnamed_addr constant

// std::nullptr_t
// CHECK: @_ZTIDn = unnamed_addr constant
// CHECK: @_ZTIPDn = unnamed_addr constant
// CHECK: @_ZTIPKDn = unnamed_addr constant

// bool
// CHECK: @_ZTIb = unnamed_addr constant
// CHECK: @_ZTIPb = unnamed_addr constant
// CHECK: @_ZTIPKb = unnamed_addr constant

// wchar_t
// CHECK: @_ZTIw = unnamed_addr constant
// CHECK: @_ZTIPw = unnamed_addr constant
// CHECK: @_ZTIPKw = unnamed_addr constant

// char
// CHECK: @_ZTIc = unnamed_addr constant
// CHECK: @_ZTIPc = unnamed_addr constant
// CHECK: @_ZTIPKc = unnamed_addr constant

// unsigned char
// CHECK: @_ZTIh = unnamed_addr constant
// CHECK: @_ZTIPh = unnamed_addr constant
// CHECK: @_ZTIPKh = unnamed_addr constant

// signed char
// CHECK: @_ZTIa = unnamed_addr constant
// CHECK: @_ZTIPa = unnamed_addr constant
// CHECK: @_ZTIPKa = unnamed_addr constant

// short
// CHECK: @_ZTIs = unnamed_addr constant
// CHECK: @_ZTIPs = unnamed_addr constant
// CHECK: @_ZTIPKs = unnamed_addr constant

// unsigned short
// CHECK: @_ZTIt = unnamed_addr constant
// CHECK: @_ZTIPt = unnamed_addr constant
// CHECK: @_ZTIPKt = unnamed_addr constant

// int
// CHECK: @_ZTIi = unnamed_addr constant
// CHECK: @_ZTIPi = unnamed_addr constant
// CHECK: @_ZTIPKi = unnamed_addr constant

// unsigned int
// CHECK: @_ZTIj = unnamed_addr constant
// CHECK: @_ZTIPj = unnamed_addr constant
// CHECK: @_ZTIPKj = unnamed_addr constant

// long
// CHECK: @_ZTIl = unnamed_addr constant
// CHECK: @_ZTIPl = unnamed_addr constant
// CHECK: @_ZTIPKl = unnamed_addr constant

// unsigned long
// CHECK: @_ZTIm = unnamed_addr constant
// CHECK: @_ZTIPm = unnamed_addr constant
// CHECK: @_ZTIPKm = unnamed_addr constant

// long long
// CHECK: @_ZTIx = unnamed_addr constant
// CHECK: @_ZTIPx = unnamed_addr constant
// CHECK: @_ZTIPKx = unnamed_addr constant

// unsigned long long
// CHECK: @_ZTIy = unnamed_addr constant
// CHECK: @_ZTIPy = unnamed_addr constant
// CHECK: @_ZTIPKy = unnamed_addr constant

// float
// CHECK: @_ZTIf = unnamed_addr constant
// CHECK: @_ZTIPf = unnamed_addr constant
// CHECK: @_ZTIPKf = unnamed_addr constant

// double
// CHECK: @_ZTId = unnamed_addr constant
// CHECK: @_ZTIPd = unnamed_addr constant
// CHECK: @_ZTIPKd = unnamed_addr constant

// long double
// CHECK: @_ZTIe = unnamed_addr constant
// CHECK: @_ZTIPe = unnamed_addr constant
// CHECK: @_ZTIPKe = unnamed_addr constant

// char16_t
// CHECK: @_ZTIDs = unnamed_addr constant
// CHECK: @_ZTIPDs = unnamed_addr constant
// CHECK: @_ZTIPKDs = unnamed_addr constant

// char32_t
// CHECK: @_ZTIDi = unnamed_addr constant
// CHECK: @_ZTIPDi = unnamed_addr constant
// CHECK: @_ZTIPKDi = unnamed_addr constant

