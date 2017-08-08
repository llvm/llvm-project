
struct AClass {
  int method();
  int method2();
};
struct AWrapperClass {
  AClass &object();
};

void takesClass(AWrapperClass &wrapper) {
  wrapper.object().method();
  wrapper.object().method();
  wrapper.object().method2();
}
// CHECK1: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-4]]:3
// CHECK2: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-4]]:3
// CHECK3: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-4]]:3

// Suggest extracting 'wrapper.object()'
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:11:3-19 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:12:3-19 %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:13:3-19 %s | FileCheck --check-prefix=CHECK3 %s

// CHECK-NO: Failed to initiate the refactoring action!

// Avoid suggesting extraction of 'wrapper.object().method2()'
// RUN: not clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:13:20-30 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// RUN: clang-refactor-test list-actions -at=%s:11:11 %s | FileCheck --check-prefix=CHECK-ACTION %s
// CHECK-ACTION: Extract Repeated Expression

AClass &returnsReference(int x);
AClass &returnsAndTakesFunctionPointer(AClass& (*)(int));

void checkReferenceCall() {
  returnsReference(0).method();
  returnsReference(0).method2();
  returnsAndTakesFunctionPointer(returnsReference).method();
  returnsAndTakesFunctionPointer(returnsReference).method2();
}
// CHECK4: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-5]]:3
// CHECK5: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-4]]:3
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:36:3-22 %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:38:3-51 %s | FileCheck --check-prefix=CHECK5 %s
// RUN: not clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:36:23-32 -in=%s:37:23-32 -in=%s:38:52-61 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

struct WithFields {
  int x, y;
};
struct WithFieldsOperators {
  WithFields *operator ->();
  WithFields &operator ()();

  const WithFields &operator [](int x) const;
  WithFields &operator [](int x);
};

void checkOperatorCalls(WithFieldsOperators &op, int id) {
  op[id].x;
  op[id].y;
// CHECK6: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-2]]:3
  op().x;
  op().x;
// CHECK7: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-2]]:3
  op->x;
  op->x;
}
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:59:3-9 %s | FileCheck --check-prefix=CHECK6 %s
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:62:3-7 %s | FileCheck --check-prefix=CHECK7 %s
// RUN: not clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:59:10-12 -in=%s:62:8-10 -in=%s:65:3-9 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

struct AWrapperClass2 {
  AClass *object() const;
};

void checkPointerType(AWrapperClass *object, AWrapperClass2 *object2) {
  object->object().method();
  if (object) {
    object->object().method2();
  }
// CHECK8: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-4]]:3
  object2->object()->method();
  int m = object2->object()->method2();
// CHECK9: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-2]]:3
}
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:77:3-19 %s | FileCheck --check-prefix=CHECK8 %s
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:82:3-20 %s | FileCheck --check-prefix=CHECK9 %s

struct ConstVsNonConst {
  int field;
  void constMethod() const;
  void method();
};

struct ConstVsNonConstWrapper {
  const ConstVsNonConst &object() const;
  ConstVsNonConst &object();
};

void checkFoo(ConstVsNonConstWrapper &object) {
  object.object().constMethod();
  object.object().method();
}
// CHECK10: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-3]]:3
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -at=%s:101:3 %s | FileCheck --check-prefix=CHECK10 %s

// Check that the action can be initiate using a selection:
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -selected=%s:11:3-11:19 -selected=%s:11:15-11:19 -selected=%s:11:3-11:17 -selected=%s:11:15-11:18 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: not clang-refactor-test initiate -action extract-repeated-expr-into-var -selected=%s:11:3-11:22 -selected=%s:11:1-13:30 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
