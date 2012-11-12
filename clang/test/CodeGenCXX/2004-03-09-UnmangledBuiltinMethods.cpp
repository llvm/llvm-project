// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// CHECK: _ZN11AccessFlags6strlenEv
struct AccessFlags {
  void strlen();
};

void AccessFlags::strlen() { }
