/// Check that std::tm and a few others are mangled as tm on Solaris only.
/// Issue #33114.
///
// RUN: %clang_cc1 -emit-llvm %s -o - -triple amd64-pc-solaris2.11 | FileCheck --check-prefix=CHECK-SOLARIS %s
// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-unknown-linux-gnu  | FileCheck --check-prefix=CHECK-LINUX %s
//
// REQUIRES: x86-registered-target
//
// CHECK-SOLARIS: @_Z6tmfunc2tm
// CHECK-SOLARIS: @_Z7ldtfunc6ldiv_t
// CHECK-LINUX:   @_Z6tmfuncSt2tm
// CHECK-LINUX:   @_Z7ldtfuncSt6ldiv_t

namespace std {
  extern "C" {
    struct tm {
      int tm_sec;
    };
    struct ldiv_t {
      long quot;
    };
  }
}

void tmfunc (std::tm tm) {}

void ldtfunc (std::ldiv_t ldt) {}
