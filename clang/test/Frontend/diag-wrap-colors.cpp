// RUN: not %clang_cc1 %s -fmessage-length=50 -fcolor-diagnostics -fno-show-source-location -o - 2>&1 | FileCheck %s

struct F {
  float a : 10;
};
// CHECK: bit-field 'a' has non-integral type 'float'
