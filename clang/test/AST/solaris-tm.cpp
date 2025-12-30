/// Check that std::tm and a few others are mangled as tm on Solaris only.
/// Issue #33114.
///
// RUN: %clang_cc1 -emit-llvm %s -o - -triple amd64-pc-solaris2.11 | FileCheck --check-prefix=CHECK-SOLARIS %s
// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-unknown-linux-gnu  | FileCheck --check-prefix=CHECK-LINUX %s
//
// REQUIRES: x86-registered-target

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

// CHECK-SOLARIS: @_Z6tmfunc2tm
// CHECK-SOLARIS: @_Z9tmccpfunc2tmPKcS1_
// CHECK-SOLARIS: @_Z7tm2func2tmS_
// CHECK-LINUX:   @_Z6tmfuncSt2tm
// CHECK-LINUX:   @_Z9tmccpfuncSt2tmPKcS1_
// CHECK-LINUX:   @_Z7tm2funcSt2tmS_

void tmfunc (std::tm tm) {}
void tmccpfunc (std::tm tm, const char *ccp, const char *ccp2) {}
void tm2func (std::tm tm, std::tm tm2) {}

// CHECK-SOLARIS: @_Z7ldtfunc6ldiv_t
// CHECK-LINUX:   @_Z7ldtfuncSt6ldiv_t

void ldtfunc (std::ldiv_t ldt) {}
