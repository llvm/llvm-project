// RUN: %clang_cc1 -triple thumbv7--windows-msvc -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple thumbv7--windows-itanium -fno-use-cxa-atexit -emit-llvm -o - -x c++ %s | FileCheck %s

class C {
public:
  ~C();
};

static C sc;
void f(const C &ci) { sc = ci; }

// CHECK: atexit

