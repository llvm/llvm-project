// RUN: %clang_cc1  -emit-llvm -g %s -o %t
// RUN: grep DW_TAG_enumeration_type %t
// Radar 8195980

enum vtag {
  VT_ONE
};

int foo(int i) {
  return i == VT_ONE;
}
