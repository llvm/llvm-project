// RUN: %clang_cc1 -std=c++11 -fsanitize=signed-integer-overflow,integer-divide-by-zero,float-divide-by-zero,shift-base,shift-exponent,unreachable,return,vla-bound,alignment,null,vptr,object-size,float-cast-overflow,bool,enum,array-bounds,function -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s

bool GetOptionalBool(bool *value);
bool GetBool(bool default_value) {
  // CHECK-LABEL: @_Z7GetBoolb
  // CHECK-NOT: select
  bool value;
  return GetOptionalBool(&value) ? value : default_value;
}
