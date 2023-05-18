// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -o - %s | FileCheck %s

// CHECK: _ZN11AccessFlags6strlenEv
struct AccessFlags {
  void strlen();
};

void AccessFlags::strlen() { }
