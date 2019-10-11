// RUN: clang-tidy -checks='-*,modernize-use-override,clang-diagnostic-shadow,clang-diagnostic-float-conversion' %s -- | count 0
//
// Enable warnings using -config:
// RUN: clang-tidy -checks='-*,modernize-use-override,clang-diagnostic-shadow,clang-diagnostic-float-conversion' \
// RUN:   -config='{ExtraArgs: ["-Wshadow","-Wno-unused-variable"], ExtraArgsBefore: ["-Wno-shadow","-Wfloat-conversion","-Wunused-variable"]}' %s -- \
// RUN:   | FileCheck -implicit-check-not='{{warning:|error:}}' %s
//
// ... -extra-arg:
// RUN: clang-tidy -checks='-*,modernize-use-override,clang-diagnostic-shadow,clang-diagnostic-float-conversion' \
// RUN:   -extra-arg=-Wshadow -extra-arg=-Wno-unused-variable \
// RUN:   -extra-arg-before=-Wno-shadow -extra-arg-before=-Wfloat-conversion \
// RUN:   -extra-arg-before=-Wunused-variable %s -- \
// RUN:   | FileCheck -implicit-check-not='{{warning:|error:}}' %s
//
// ... a combination of -config and -extra-arg(-before):
// RUN: clang-tidy -checks='-*,modernize-use-override,clang-diagnostic-shadow,clang-diagnostic-float-conversion' \
// RUN:   -config='{ExtraArgs: ["-Wno-unused-variable"], ExtraArgsBefore: ["-Wno-shadow","-Wfloat-conversion"]}' \
// RUN:   -extra-arg=-Wshadow -extra-arg-before=-Wunused-variable %s -- \
// RUN:   | FileCheck -implicit-check-not='{{warning:|error:}}' %s

void f(float x) {
  int a;
  { int a; }
  // CHECK: :[[@LINE-1]]:9: warning: declaration shadows a local variable [clang-diagnostic-shadow]
  int b = x;
  // CHECK: :[[@LINE-1]]:11: warning: implicit conversion turns floating-point number into integer: 'float' to 'int' [clang-diagnostic-float-conversion]
}
