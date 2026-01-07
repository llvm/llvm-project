// RUN: %clang_cc1 %s -std=c++11 -triple=i686-pc-windows-msvc -debug-info-kind=limited -gcodeview -emit-llvm -o - | FileCheck %s

// The injected class names in this test were accidentally making it into our
// nested class record debug info. Make sure they don't appear there.

// PR28790

struct A {
  const char *m_fn1();
  template <typename> class B;
  template <typename> class C;
  template <typename FunctionIdT> class C<B<FunctionIdT>>;
};
const char *A::m_fn1() { return nullptr; }

// CHECK: ![[A:[^ ]*]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A",
// CHECK-SAME: elements: ![[elements:[0-9]+]]

// CHECK: ![[elements]] = !{![[m_fn1:[0-9]+]]}

// CHECK: ![[m_fn1]] = !DISubprogram(name: "m_fn1",
