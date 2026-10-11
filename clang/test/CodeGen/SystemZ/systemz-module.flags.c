// REQUIRES: clang-target-64-bits

// RUN: %clang_cc1 -source-date-epoch 253402300799 -triple s390x-ibm-zos -emit-llvm -o - %s | FileCheck %s
// CHECK: {{.*}}"zos_cu_language", !"C"}
// CHECK: {{.*}}"zos_translation_time", i64 253402300799}
