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
// CHECK: @_ZTIv = constant
// CHECK: @_ZTIPv = constant
// CHECK: @_ZTIPKv = constant

// std::nullptr_t
// CHECK: @_ZTIDn = constant
// CHECK: @_ZTIPDn = constant
// CHECK: @_ZTIPKDn = constant

// bool
// CHECK: @_ZTIb = constant
// CHECK: @_ZTIPb = constant
// CHECK: @_ZTIPKb = constant

// wchar_t
// CHECK: @_ZTIw = constant
// CHECK: @_ZTIPw = constant
// CHECK: @_ZTIPKw = constant

// char
// CHECK: @_ZTIc = constant
// CHECK: @_ZTIPc = constant
// CHECK: @_ZTIPKc = constant

// unsigned char
// CHECK: @_ZTIh = constant
// CHECK: @_ZTIPh = constant
// CHECK: @_ZTIPKh = constant

// signed char
// CHECK: @_ZTIa = constant
// CHECK: @_ZTIPa = constant
// CHECK: @_ZTIPKa = constant

// short
// CHECK: @_ZTIs = constant
// CHECK: @_ZTIPs = constant
// CHECK: @_ZTIPKs = constant

// unsigned short
// CHECK: @_ZTIt = constant
// CHECK: @_ZTIPt = constant
// CHECK: @_ZTIPKt = constant

// int
// CHECK: @_ZTIi = constant
// CHECK: @_ZTIPi = constant
// CHECK: @_ZTIPKi = constant

// unsigned int
// CHECK: @_ZTIj = constant
// CHECK: @_ZTIPj = constant
// CHECK: @_ZTIPKj = constant

// long
// CHECK: @_ZTIl = constant
// CHECK: @_ZTIPl = constant
// CHECK: @_ZTIPKl = constant

// unsigned long
// CHECK: @_ZTIm = constant
// CHECK: @_ZTIPm = constant
// CHECK: @_ZTIPKm = constant

// long long
// CHECK: @_ZTIx = constant
// CHECK: @_ZTIPx = constant
// CHECK: @_ZTIPKx = constant

// unsigned long long
// CHECK: @_ZTIy = constant
// CHECK: @_ZTIPy = constant
// CHECK: @_ZTIPKy = constant

// __int128
// CHECK: @_ZTIn = constant
// CHECK: @_ZTIPn = constant
// CHECK: @_ZTIPKn = constant

// unsigned __int128
// CHECK: @_ZTIo = constant
// CHECK: @_ZTIPo = constant
// CHECK: @_ZTIPKo = constant

// half
// CHECK: @_ZTIDh = constant
// CHECK: @_ZTIPDh = constant
// CHECK: @_ZTIPKDh = constant

// float
// CHECK: @_ZTIf = constant
// CHECK: @_ZTIPf = constant
// CHECK: @_ZTIPKf = constant

// double
// CHECK: @_ZTId = constant
// CHECK: @_ZTIPd = constant
// CHECK: @_ZTIPKd = constant

// long double
// CHECK: @_ZTIe = constant
// CHECK: @_ZTIPe = constant
// CHECK: @_ZTIPKe = constant

// char16_t
// CHECK: @_ZTIDs = constant
// CHECK: @_ZTIPDs = constant
// CHECK: @_ZTIPKDs = constant

// char32_t
// CHECK: @_ZTIDi = constant
// CHECK: @_ZTIPDi = constant
// CHECK: @_ZTIPKDi = constant

