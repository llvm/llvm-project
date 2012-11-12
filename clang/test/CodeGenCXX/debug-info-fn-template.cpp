// RUN: %clang -emit-llvm -g -S %s -o - | FileCheck %s

template<typename T>
struct XF {
  T member;
};

template<typename T>
T fx(XF<T> xi) {
  return xi.member;
}

//CHECK: XF<int>
//CHECK: DW_TAG_template_type_parameter
template int fx(XF<int>);
