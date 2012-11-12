// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodule-cache-path %t -I %S/Inputs -verify %s -Wno-objc-root-class
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodule-cache-path %t -I %S/Inputs -emit-llvm %s -o - -Wno-objc-root-class | grep Emit | FileCheck %s
// expected-no-diagnostics

@__experimental_modules_import templates_left;
@__experimental_modules_import templates_right;


void testTemplateClasses() {
  Vector<int> vec_int;
  vec_int.push_back(0);

  List<bool> list_bool;
  list_bool.push_back(false);

  N::Set<char> set_char;
  set_char.insert('A');
}

void testPendingInstantiations() {
  // CHECK: call {{.*pendingInstantiationEmit}}
  // CHECK: call {{.*pendingInstantiationEmit}}
  // CHECK: define {{.*pendingInstantiationEmit.*[(]i}}
  // CHECK: define {{.*pendingInstantiationEmit.*[(]double}}
  triggerPendingInstantiation();
  triggerPendingInstantiationToo();
}

void testRedeclDefinition() {
  // CHECK: define {{.*redeclDefinitionEmit}}
  redeclDefinitionEmit();
}

// CHECK: call {{.*pendingInstantiation}}
// CHECK: call {{.*redeclDefinitionEmit}}
