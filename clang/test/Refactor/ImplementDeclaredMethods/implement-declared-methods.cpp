
struct Class {
  int field;

  Class();

  Class(int x) { }

  ~Class();

  // commment middle-methods-begin: +1:1
  static void method(const int &value, int defaultParam = 20);

  virtual int voidMethod(int y) const;
  void implementedMethod() const { // middle-methods-end: -1:40

  }

  void outOfLineImpl(int x);

  void anotherImplementedMethod() {

  }
};
// CHECK1: "{{.*}}implement-declared-methods.cpp" "\n\nvoid Class::method(const int &value, int defaultParam) { \n  <#code#>;\n}\n\nint Class::voidMethod(int y) const { \n  <#code#>;\n}\n" [[@LINE+5]]:37 -> [[@LINE+5]]:37
// CHECK2: "{{.*}}implement-declared-methods.cpp" "\n\nClass::Class() { \n  <#code#>;\n}\n\nClass::~Class() { \n  <#code#>;\n}\n\nvoid Class::method(const int &value, int defaultParam) { \n  <#code#>;\n}\n\nint Class::voidMethod(int y) const { \n  <#code#>;\n}\n"  [[@LINE+4]]:37
// CHECK3: "{{.*}}implement-declared-methods.cpp" "\n\nClass::~Class() { \n  <#code#>;\n}\n\nvoid Class::method(const int &value, int defaultParam) { \n  <#code#>;\n}\n" [[@LINE+3]]:37
// CHECK4: "{{.*}}implement-declared-methods.cpp" "\n\nClass::Class() { \n  <#code#>;\n}\n\nvoid Class::method(const int &value, int defaultParam) { \n  <#code#>;\n}\n" [[@LINE+2]]:37

void Class::outOfLineImpl(int x) { }

// query-all-impl: [ { name: ast.producer.query, filenameResult: "%s" } , { name: decl.query , predicateResults: [{name: decl.isDefined, intValues: [0, 0, 0, 0] }] }]
// RUN: clang-refactor-test perform -action implement-declared-methods -selected=middle-methods -continuation-file=%s -query-results=query-all-impl %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action implement-declared-methods -selected=%s:5:1-20:1 -continuation-file=%s -query-results=query-all-impl %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test perform -action implement-declared-methods -selected=%s:8:1-12:10 -continuation-file=%s -query-results=query-all-impl %s | FileCheck --check-prefix=CHECK3 %s

// Implement the constructor and method:
// query-mix-impl: [ { name: ast.producer.query, filenameResult: "%s" } , { name: decl.query , predicateResults: [{name: decl.isDefined, intValues: [0, 1, 0, 1] }] }]
// RUN: clang-refactor-test perform -action implement-declared-methods -selected=%s:5:1-20:1 -continuation-file=%s -query-results=query-mix-impl %s | FileCheck --check-prefix=CHECK4 %s

// RUN: clang-refactor-test perform -action implement-declared-methods -selected=%s:5:1-20:1 -continuation-file=%S/Inputs/class.cpp -query-results=query-mix-impl %s | FileCheck --check-prefix=CHECK1 %S/Inputs/class.cpp

// Empty continuation TU should produce an error:
// RUN: not clang-refactor-test perform -action implement-declared-methods -selected=%s:5:1-20:1 -continuation-file=%S/Inputs/empty.cpp -query-results=query-mix-impl %s 2>&1 | FileCheck --check-prefix=CHECK-EMPTY-ERR %s
// CHECK-EMPTY-ERR: failed to perform the refactoring continuation (the target class is not defined in the continuation AST unit)!

#ifdef USE_NAMESPACE
namespace ns {
namespace ns2 {
#endif

#ifdef USE_ENCLOSING_RECORD
struct OuterRecord {
#endif

struct ClassInHeader {
// class-in-header-begin: +1:1
  void pleaseImplement();
  void implemented();
  void pleaseImplementThisAsWell();
  void implementedToo();
// class-in-header-end: +1:1
};

#ifdef USE_ENCLOSING_RECORD
struct }
#endif

#ifdef USE_NAMESPACE
}
}
#endif

// RUN: clang-refactor-test perform -action implement-declared-methods -selected=class-in-header -continuation-file=%S/Inputs/classInHeader.cpp -query-results=query-mix-impl %s | FileCheck --check-prefix=CHECK1 %S/Inputs/classInHeader.cpp
// RUN: clang-refactor-test perform -action implement-declared-methods -selected=class-in-header -continuation-file=%S/Inputs/classInHeader.cpp -query-results=query-mix-impl %s -DUSE_NAMESPACE | FileCheck --check-prefix=CHECK1 %S/Inputs/classInHeader.cpp
// RUN: clang-refactor-test perform -action implement-declared-methods -selected=class-in-header -continuation-file=%S/Inputs/classInHeader.cpp -query-results=query-mix-impl %s -DUSE_NAMESPACE -DUSE_NAMESPACE_USING | FileCheck --check-prefix=CHECK1 %S/Inputs/classInHeader.cpp
// RUN: clang-refactor-test perform -action implement-declared-methods -selected=class-in-header -continuation-file=%S/Inputs/classInHeader.cpp -query-results=query-mix-impl %s -DUSE_NAMESPACE -DUSE_NAMESPACE_PREFIX | FileCheck --check-prefix=CHECK1-NS-PREFIX %S/Inputs/classInHeader.cpp

// Test when the implementation file has no out-of-line definitions.
// RUN: clang-refactor-test perform -action implement-declared-methods -selected=class-in-header -continuation-file=%S/Inputs/classInHeader.cpp -query-results=query-mix-impl %s -DNO_IMPL | FileCheck --check-prefix=CHECK1-NO-IMPL %S/Inputs/classInHeader.cpp
// RUN: clang-refactor-test perform -action implement-declared-methods -selected=class-in-header -continuation-file=%S/Inputs/classInHeader.cpp -query-results=query-mix-impl %s -DNO_IMPL -DUSE_NAMESPACE | FileCheck --check-prefix=CHECK1-NO-IMPL-USING-NS %S/Inputs/classInHeader.cpp
// RUN: clang-refactor-test perform -action implement-declared-methods -selected=class-in-header -continuation-file=%S/Inputs/classInHeader.cpp -query-results=query-mix-impl %s -DNO_IMPL -DUSE_NAMESPACE -DUSE_ENCLOSING_RECORD | FileCheck --check-prefix=CHECK1-NO-IMPL-USING-NS-IN-RECORD %S/Inputs/classInHeader.cpp
// RUN: clang-refactor-test perform -action implement-declared-methods -selected=class-in-header -continuation-file=%S/Inputs/classInHeader.cpp -query-results=query-mix-impl %s -DNO_IMPL -DUSE_NAMESPACE -DUSE_NAMESPACE_USING | FileCheck --check-prefix=CHECK1-NO-IMPL %S/Inputs/classInHeader.cpp

// query-mix-impl-header: [ { name: ast.producer.query, filenameResult: "%S/classInHeader.h" } , { name: decl.query , predicateResults: [{name: decl.isDefined, intValues: [0, 1, 0, 1] }] }]
// RUN: clang-refactor-test perform -action implement-declared-methods -selected=class-in-header -continuation-file=%S/Inputs/classInHeader.cpp -query-results=query-mix-impl-header %s | FileCheck %S/Inputs/classInHeader.h

// query-no-impl: [ { name: ast.producer.query, filenameResult: "%s" } , { name: decl.query , predicateResults: [{name: decl.isDefined, intValues: [1, 1, 1, 1] }] }]
// RUN: not clang-refactor-test perform -action implement-declared-methods -selected=class-in-header -continuation-file=%S/Inputs/classInHeader.cpp -query-results=query-no-impl %s 2>&1 | FileCheck --check-prefix=ALL-IMPLEMENTED-ERROR %s
// ALL-IMPLEMENTED-ERROR: error: continuation failed: the selected methods are already implemented

// Ensure that the methods which are placed right after the record are placed
// after the outermost record:
namespace ns {

struct AfterRecordOuterOuter {
struct AfterRecordOuter {
  struct AfterRecordInner {
// after-record-inner-begin: +1:1
    void pleaseImplement();
// after-record-inner-end: +0:1
  };

  AfterRecordOuter();
};
// comment
};
// CHECK-OUTERMOST: "{{.*}}implement-declared-methods.cpp" "\n\nvoid AfterRecordOuterOuter::AfterRecordOuter::AfterRecordInner::pleaseImplement() { \n  <#code#>;\n}\n" [[@LINE-1]]:3 -> [[@LINE-1]]:3

}
// RUN: clang-refactor-test perform -action implement-declared-methods -selected=after-record-inner -continuation-file=%s -query-results=query-all-impl %s | FileCheck --check-prefix=CHECK-OUTERMOST %s

#ifdef INNER_TEMPLATE
template<typename T>
struct OuterTemplateRecord {
#else
  template<typename U>
#endif
  struct InnerTemplate {
// inner-template-begin: +0:1
    InnerTemplate();
    void function();
// inner-template-end: +1:1
  };
#ifdef INNER_TEMPLATE
};
#endif

// RUN: not clang-refactor-test perform -action implement-declared-methods -selected=inner-template %s 2>&1 | FileCheck --check-prefix=CHECK-TEMPLATE-NO %s
// RUN: not clang-refactor-test perform -action implement-declared-methods -selected=inner-template %s -DINNER_TEMPLATE 2>&1 | FileCheck --check-prefix=CHECK-TEMPLATE-NO %s

// CHECK-TEMPLATE-NO: Failed to initiate the refactoring action (templates are unsupported)!

template<int x, typename T>
class TemplateSpecialization {
};

template<>
class TemplateSpecialization<0, int> {
// template-specialization-begin: +0:1
  TemplateSpecialization();
  void function();
  void operator ()(int) const;
  operator int() const;
// template-specialization-end: +0:1
};
// CHECK-SPECIALIZATION: "{{.*}}implement-declared-methods.cpp" "\n\nTemplateSpecialization<0, int>::TemplateSpecialization() { \n  <#code#>;\n}\n\nvoid TemplateSpecialization<0, int>::function() { \n  <#code#>;\n}\n\nvoid TemplateSpecialization<0, int>::operator()(int) const { \n  <#code#>;\n}\n\nTemplateSpecialization<0, int>::operator int() const { \n  <#code#>;\n}\n" [[@LINE-1]]:3
// RUN: clang-refactor-test perform -action implement-declared-methods -selected=template-specialization -continuation-file=%s -query-results=query-all-impl %s | FileCheck --check-prefix=CHECK-SPECIALIZATION %s

template<int x>
class TemplateSpecialization<x, int> {
// template-partial-specialization-begin: +0:1
  void function();
// template-partial-specialization-end: +0:1
};

// RUN: not clang-refactor-test perform -action implement-declared-methods -selected=template-partial-specialization %s -DINNER_TEMPLATE 2>&1 | FileCheck --check-prefix=CHECK-TEMPLATE-NO %s

struct ProhibitTemplateFunctions {
// template-function-begin: +0:1
  void function();
  template<typename T>
  void functionTemplate(const T &);
  void anotherFunction();
// template-function-end: +0:1
};
// CHECK-FUNCTION-TEMPLATE: "{{.*}}implement-declared-methods.cpp" "\n\nvoid ProhibitTemplateFunctions::function() { \n  <#code#>;\n}\n\nvoid ProhibitTemplateFunctions::anotherFunction() { \n  <#code#>;\n}\n" [[@LINE-1]]:3
// RUN: clang-refactor-test perform -action implement-declared-methods -selected=template-function -continuation-file=%s -query-results=query-all-impl %s | FileCheck --check-prefix=CHECK-FUNCTION-TEMPLATE %s
