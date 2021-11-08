// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck -check-prefix CHECK-PRECISE %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -menable-no-nans -emit-llvm -o - %s | FileCheck -check-prefix CHECK-NO-NANS %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -menable-no-infs -emit-llvm -o - %s | FileCheck -check-prefix CHECK-NO-INFS %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ffinite-math-only -emit-llvm -o - %s | FileCheck -check-prefix CHECK-FINITE %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fno-signed-zeros -emit-llvm -o - %s | FileCheck -check-prefix CHECK-NO-SIGNED-ZEROS %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -mreassociate -emit-llvm -o - %s | FileCheck -check-prefix CHECK-REASSOC %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -freciprocal-math -emit-llvm -o - %s | FileCheck -check-prefix CHECK-RECIP %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -menable-unsafe-fp-math -emit-llvm -o - %s | FileCheck -check-prefix CHECK-UNSAFE %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ffast-math -emit-llvm -o - %s | FileCheck -check-prefix CHECK-FAST %s

float fn(float);

float test(float a) {
  return a + fn(a);
}

// CHECK-PRECISE: [[CALL_RES:%.+]] = call float @fn(float noundef {{%.+}})
// CHECK-PRECISE: {{%.+}} = fadd float {{%.+}}, [[CALL_RES]]

// CHECK-NO-NANS: [[CALL_RES:%.+]] = call nnan float @fn(float noundef {{%.+}})
// CHECK-NO-NANS: {{%.+}} = fadd nnan float {{%.+}}, [[CALL_RES]]

// CHECK-NO-INFS: [[CALL_RES:%.+]] = call ninf float @fn(float noundef {{%.+}})
// CHECK-NO-INFS: {{%.+}} = fadd ninf float {{%.+}}, [[CALL_RES]]

// CHECK-FINITE: [[CALL_RES:%.+]] = call nnan ninf float @fn(float noundef {{%.+}})
// CHECK-FINITE: {{%.+}} = fadd nnan ninf float {{%.+}}, [[CALL_RES]]

// CHECK-NO-SIGNED-ZEROS: [[CALL_RES:%.+]] = call nsz float @fn(float noundef {{%.+}})
// CHECK-NO-SIGNED-ZEROS: {{%.+}} = fadd nsz float {{%.+}}, [[CALL_RES]]

// CHECK-REASSOC: [[CALL_RES:%.+]] = call reassoc float @fn(float noundef {{%.+}})
// CHECK-REASSOC: {{%.+}} = fadd reassoc float {{%.+}}, [[CALL_RES]]

// CHECK-RECIP: [[CALL_RES:%.+]] = call arcp float @fn(float noundef {{%.+}})
// CHECK-RECIP: {{%.+}} = fadd arcp float {{%.+}}, [[CALL_RES]]

// CHECK-UNSAFE: [[CALL_RES:%.+]] = call reassoc nsz arcp afn float @fn(float noundef {{%.+}})
// CHECK-UNSAFE: {{%.+}} = fadd reassoc nsz arcp afn float {{%.+}}, [[CALL_RES]]

// CHECK-FAST: [[CALL_RES:%.+]] = call reassoc nnan ninf nsz arcp afn float @fn(float noundef {{%.+}})
// CHECK-FAST: {{%.+}} = fadd reassoc nnan ninf nsz arcp afn float {{%.+}}, [[CALL_RES]]
