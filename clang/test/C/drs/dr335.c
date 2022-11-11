/* RUN: %clang_cc1 -std=c89 -triple x86_64-pc-win32 -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c89 -triple i686-pc-linux -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c99 -triple x86_64-pc-win32 -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c99 -triple i686-pc-linux -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c11 -triple x86_64-pc-win32 -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c11 -triple i686-pc-linux -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c17 -triple x86_64-pc-win32 -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c17 -triple i686-pc-linux -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c2x -triple x86_64-pc-win32 -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c2x -triple i686-pc-linux -emit-llvm -o - %s | FileCheck %s
 */

/* WG14 DR335: yes
 * _Bool bit-fields
 *
 * This validates the runtime behavior from the DR, see dr3xx.c for the compile
 * time enforcement portion.
 */
void dr335(void) {
  struct bits_ {
    _Bool bbf1 : 1;
  } bits = { 1 };

  bits.bbf1 = ~bits.bbf1;

  // First, load the value from bits.bbf1 and truncate it down to one-bit.

  // CHECK: %[[LOAD1:.+]] = load i8, ptr {{.+}}, align 1
  // CHECK-NEXT: %[[CLEAR1:.+]] = and i8 %[[LOAD1]], 1
  // CHECK-NEXT: %[[CAST:.+]] = trunc i8 %[[CLEAR1]] to i1
  // CHECK-NEXT: %[[CONV:.+]] = zext i1 %[[CAST]] to i32

  // Second, perform the unary complement.

  // CHECK-NEXT: %[[NOT:.+]] = xor i32 %[[CONV]], -1

  // Finally, test the new value against 0. If it's nonzero, then assign one
  // into the bit-field, otherwise assign zero into the bit-field. Note, this
  // does not perform the operation on the promoted value, so this matches the
  // requirements in C99 6.3.1.2, so a bit-field of type _Bool behaves like a
  // _Bool and not like an [unsigned] int.
  // CHECK-NEXT: %[[TOBOOL:.+]] = icmp ne i32 %[[NOT]], 0
  // CHECK-NEXT: %[[ZERO:.+]] = zext i1 %[[TOBOOL]] to i8
  // CHECK-NEXT: %[[LOAD2:.+]] = load i8, ptr {{.+}}, align 1
  // CHECK-NEXT: %[[CLEAR2:.+]] = and i8 %[[LOAD2]], -2
  // CHECK-NEXT: %[[SET:.+]] = or i8 %[[CLEAR2]], %[[ZERO]]
  // CHECK-NEXT: store i8 %[[SET]], ptr {{.+}}, align 1
  // CHECK-NEXT: {{.+}} = trunc i8 %[[ZERO]] to i1
}

