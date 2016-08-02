// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-apple-darwin %s -o - | FileCheck %s

struct MyClass {
  template <int i> int add(int j) {
    return i + j;
  }
  virtual void func() {
  }
};

int add2(int x) {
  return MyClass().add<2>(x);
}

inline int add3(int x) {
  return MyClass().add<3>(x); // even though add<3> is ODR used, don't emit it since we don't codegen it
}

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "foo"
// CHECK-SAME:             elements: [[FOO_MEM:![0-9]*]]
// CHECK-SAME:             identifier: "_ZTS3foo"
// CHECK: [[FOO_MEM]] = !{[[FOO_FUNC:![0-9]*]]}
// CHECK: [[FOO_FUNC]] = !DISubprogram(name: "func", linkageName: "_ZN3foo4funcEN5outerIS_E5innerE",
// CHECK-SAME:                         type: [[FOO_FUNC_TYPE:![0-9]*]]
// CHECK: [[FOO_FUNC_TYPE]] = !DISubroutineType(types: [[FOO_FUNC_PARAMS:![0-9]*]])
// CHECK: [[FOO_FUNC_PARAMS]] = !{null, !{{[0-9]*}}, !"[[OUTER_FOO_INNER_ID:.*]]"}
// CHECK: !{{[0-9]*}} = !DICompositeType(tag: DW_TAG_structure_type, name: "inner"{{.*}}, identifier: "[[OUTER_FOO_INNER_ID]]")

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "virt<elem>"
// CHECK-SAME:             elements: [[VIRT_MEM:![0-9]*]]
// CHECK-SAME:             vtableHolder: !"_ZTS4virtI4elemE"
// CHECK-SAME:             templateParams: [[VIRT_TEMP_PARAM:![0-9]*]]
// CHECK-SAME:             identifier: "_ZTS4virtI4elemE"
// CHECK: [[VIRT_TEMP_PARAM]] = !{[[VIRT_T:![0-9]*]]}
// CHECK: [[VIRT_T]] = !DITemplateTypeParameter(name: "T", type: !"_ZTS4elem")

// CHECK: [[C:![0-9]*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "MyClass"
// CHECK-SAME:                             elements: [[C_MEM:![0-9]*]]
// CHECK-SAME:                             vtableHolder: !"_ZTS7MyClass"
// CHECK-SAME:                             identifier: "_ZTS7MyClass")
// CHECK: [[C_MEM]] = !{[[C_VPTR:![0-9]*]], [[C_FUNC:![0-9]*]]}
// CHECK: [[C_VPTR]] = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$MyClass"

// CHECK: [[C_FUNC]] = !DISubprogram(name: "func",{{.*}} line: 7,

// CHECK: [[ELEM:![0-9]*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "elem"
// CHECK-SAME:                                elements: [[ELEM_MEM:![0-9]*]]
// CHECK-SAME:                                identifier: "_ZTS4elem"
// CHECK: [[ELEM_MEM]] = !{[[ELEM_X:![0-9]*]]}
// CHECK: [[ELEM_X]] = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !"_ZTS4elem"
// CHECK-SAME:                        baseType: !"_ZTS4virtI4elemE"

// Check that the member function template specialization and implicit special
// members (the default ctor) refer to their class by scope, even though they
// didn't appear in the class's member list (C_MEM). This prevents the functions
// from being added to type units, while still appearing in the type
// declaration/reference in the compile unit.
// CHECK: !DISubprogram(name: "MyClass"
// CHECK-SAME:          scope: !"_ZTS7MyClass"
// CHECK: !DISubprogram(name: "add<2>"
// CHECK-SAME:          scope: !"_ZTS7MyClass"

template<typename T>
struct outer {
  struct inner {
    int i;
  };
};

struct foo {
  void func(outer<foo>::inner);
};

inline void func() {
  // require 'foo' to be complete before the emission of 'inner' so that, when
  // constructing the context chain for 'x' we emit the full definition of
  // 'foo', which requires the definition of 'inner' again
  foo f;
}

outer<foo>::inner x;

// CHECK: !DIGlobalVariable(name: "x",
// CHECK-SAME:              type: !"[[OUTER_FOO_INNER_ID]]"
// CHECK-SAME:              variable: %"struct.outer<foo>::inner"* @x

template <typename T>
struct virt {
  T* values;
  virtual ~virt();
};
struct elem {
  static virt<elem> x; // ensure that completing 'elem' will require/completing 'virt<elem>'
};
inline void f1() {
  elem e; // ensure 'elem' is required to be complete when it is emitted as a template argument for 'virt<elem>'
};
void f2() {
  virt<elem> d; // emit 'virt<elem>'
}
