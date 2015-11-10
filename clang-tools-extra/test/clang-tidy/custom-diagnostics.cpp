// RUN: clang-tidy -checks='-*,modernize-use-override,clang-diagnostic-shadow,clang-diagnostic-float-conversion' %s -- | count 0
// RUN: clang-tidy -checks='-*,modernize-use-override,clang-diagnostic-shadow,clang-diagnostic-float-conversion' \
// RUN:   -config='{ExtraArgs: ["-Wshadow","-Wno-unused-variable"], ExtraArgsBefore: ["-Wno-shadow","-Wfloat-conversion","-Wunused-variable"]}' %s -- \
// RUN:   | FileCheck -implicit-check-not='{{warning:|error:}}' %s

void f(float x) {
  int a;
  { int a; }
  // CHECK: :[[@LINE-1]]:9: warning: declaration shadows a local variable [clang-diagnostic-shadow]
  int b = x;
  // CHECK: :[[@LINE-1]]:11: warning: implicit conversion turns floating-point number into integer: 'float' to 'int' [clang-diagnostic-float-conversion]
}
