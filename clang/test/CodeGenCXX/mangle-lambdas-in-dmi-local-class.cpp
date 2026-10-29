// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -emit-llvm -I%S -std=c++20 -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclang-abi-compat=22 %s -emit-llvm -I%S -std=c++20 -o - | FileCheck --check-prefix=ABICOMPAT22 %s

// Ensure that local classes mangled with <local-name> while mangling a lamda
// in default member initializer.

// Itanium ABI 5.1.2
// Names
// <...> Entities declared within a function, including members of local classes, are mangled with <local-name>. Entities declared in a namespace or class scope are mangled with <nested-name>. <...>

#include <typeinfo>

void foo() {
  {
    struct Test {
      const std::type_info& a = typeid([] {});
      const std::type_info& b = typeid([] {});
    } T;
  }

  {
    struct Test {
      const std::type_info& a = typeid([] {});
      const std::type_info& b = typeid([] {});
    } T;
  }
}

// _Z
// TI
// Z3foovE         ; local-name scope
//   N
//     4Test
//       1a        ; member a
//         M       ; member-initializer context
//           UlvE_ ; lambda closure type
//   E

// CHECK: @_ZTIZ3foovEN4Test1aMUlvE_E
// CHECK: @_ZTSZ3foovEN4Test1aMUlvE_E
// CHECK: @_ZTIZ3foovEN4Test1bMUlvE_E
// CHECK: @_ZTSZ3foovEN4Test1bMUlvE_E

// _Z
// TI
// Z3foovE         ; local-name scope
//   N
//     4Test
//       1a        ; member a
//         M       ; member-initializer context
//           UlvE_ ; lambda closure type
//   E
//   _0            ; second local 'Test' definition

// CHECK: @_ZTIZ3foovEN4Test1aMUlvE_E_0
// CHECK: @_ZTSZ3foovEN4Test1aMUlvE_E_0
// CHECK: @_ZTIZ3foovEN4Test1bMUlvE_E_0
// CHECK: @_ZTSZ3foovEN4Test1bMUlvE_E_0

// CHECK-LABEL: define internal void @_ZZ3foovEN4TestC2Ev
// CHECK: store ptr @_ZTIZ3foovEN4Test1aMUlvE_E
// CHECK: store ptr @_ZTIZ3foovEN4Test1bMUlvE_E

// CHECK-LABEL: define internal void @_ZZ3foovEN4TestC2E_0v
// CHECK: store ptr @_ZTIZ3foovEN4Test1aMUlvE_E_0
// CHECK: store ptr @_ZTIZ3foovEN4Test1bMUlvE_E_0

// ABICOMPAT22: @_ZTIN3foo4Test1aMUlvE_E
// ABICOMPAT22: @_ZTSN3foo4Test1aMUlvE_E
// ABICOMPAT22: @_ZTIN3foo4Test1bMUlvE_E
// ABICOMPAT22: @_ZTSN3foo4Test1bMUlvE_E

// ABICOMPAT22-LABEL: define internal void @_ZZ3foovEN4TestC2Ev
// ABICOMPAT22: store ptr @_ZTIN3foo4Test1aMUlvE_E
// ABICOMPAT22: store ptr @_ZTIN3foo4Test1bMUlvE_E

// ABICOMPAT22-LABEL: define internal void @_ZZ3foovEN4TestC2E_0v
// ABICOMPAT22: store ptr @_ZTIN3foo4Test1aMUlvE_E
// ABICOMPAT22: store ptr @_ZTIN3foo4Test1bMUlvE_E
