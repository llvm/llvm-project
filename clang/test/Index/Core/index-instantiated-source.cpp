// RUN: c-index-test core -print-source-symbols -- %s -std=c++14 -target x86_64-apple-macosx10.7 | FileCheck %s
// References to declarations in instantiations should be canonicalized:

template<typename T>
class BaseTemplate {
public:
  T baseTemplateFunction();
// CHECK: [[@LINE-1]]:5 | instance-method/C++ | baseTemplateFunction | c:@ST>1#T@BaseTemplate@F@baseTemplateFunction#

  T baseTemplateField;
// CHECK: [[@LINE-1]]:5 | field/C++ | baseTemplateField | c:@ST>1#T@BaseTemplate@FI@baseTemplateField
};

template<typename T, typename S>
class TemplateClass: public BaseTemplate<T> {
public:
  T function() { return T(); }
// CHECK: [[@LINE-1]]:5 | instance-method/C++ | function | c:@ST>2#T#T@TemplateClass@F@function#

  static void staticFunction() { }
// CHECK: [[@LINE-1]]:15 | static-method/C++ | staticFunction | c:@ST>2#T#T@TemplateClass@F@staticFunction#S

  T field;
// CHECK: [[@LINE-1]]:5 | field/C++ | field | c:@ST>2#T#T@TemplateClass@FI@field
};

void canonicalizeInstaniationReferences(TemplateClass<int, float> &object) {
  (void)object.function();
// CHECK: [[@LINE-1]]:16 | instance-method/C++ | function | c:@ST>2#T#T@TemplateClass@F@function# | <no-cgname>
  (void)object.field;
// CHECK: [[@LINE-1]]:16 | field/C++ | field | c:@ST>2#T#T@TemplateClass@FI@field | <no-cgname> | Ref,RelCont | rel: 1
  (void)object.baseTemplateFunction();
// CHECK: [[@LINE-1]]:16 | instance-method/C++ | baseTemplateFunction | c:@ST>1#T@BaseTemplate@F@baseTemplateFunction# | <no-cgname>
  (void)object.baseTemplateField;
// CHECK: [[@LINE-1]]:16 | field/C++ | baseTemplateField | c:@ST>1#T@BaseTemplate@FI@baseTemplateField | <no-cgname> | Ref,RelCont | rel: 1

  TemplateClass<int, float>::staticFunction();
// CHECK: [[@LINE-1]]:30 | static-method/C++ | staticFunction | c:@ST>2#T#T@TemplateClass@F@staticFunction#S | <no-cgname
}
