// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm %s -o -  | FileCheck %s

namespace Common {
enum RenderMode {
  kRenderEGA,
  kRenderCGA
};
class C;
class A {
  A();
  C *_vm;
  unsigned char _highlightColorTableVGA[];
  static const unsigned char b[];
};
// CHECK: [[Common_A_b:@[^ ]+]] = {{(dso_local )?}}constant [1 x i8] zeroinitializer
class B {
public:
  Common::RenderMode _configRenderMode;
};
class C : public B {};
A::A() {
  0 == Common::kRenderCGA || _vm->_configRenderMode == Common::kRenderEGA
      ? b
      : _highlightColorTableVGA;
// Make sure the PHI value is casted correctly to the PHI type
// CHECK: %{{.*}} = phi [0 x i8]* [ bitcast ([1 x i8]* [[Common_A_b]] to [0 x i8]*), %{{.*}} ], [ %{{.*}}, %{{.*}} ]
}
const unsigned char A::b[] = { 0 };
}
