struct AbstractClass {
  virtual void method() = 0;
  virtual void otherMethod() { }
};

struct Base {
  virtual void nonAbstractClassMethod() { }
};

struct Target : Base, AbstractClass {
  int field = 0;

  union SubRecord {
  };

  void outerMethod() const;

  void innerMethod() {
    int x = 0;
  }
};
// CHECK1: Initiated the 'fill-in-missing-abstract-methods' action at [[@LINE-12]]:1

// RUN: clang-refactor-test initiate -action fill-in-missing-abstract-methods -in=%s:10:1-end -in=%s:11:1-end -in=%s:12:1-end -in=%s:13:1-2 -in=%s:15:1-end -in=%s:16:1-2 -in=%s:17:1-end -in=%s:18:1-2 -in=%s:21:1-2 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: not clang-refactor-test initiate -action fill-in-missing-abstract-methods -in=%s:13:3-end -in=%s:14:1-2 -in=%s:16:3-27 -in=%s:18:3-end -in=%s:19:1-end -in=%s:20:1-3 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// CHECK-NO: Failed to initiate the refactoring action

// RUN: clang-refactor-test list-actions -at=%s:10:1 %s | FileCheck --check-prefix=CHECK-ACTION %s
// CHECK-ACTION: Add Missing Abstract Class Overrides


void Target::outerMethod() const {
}

// RUN: not clang-refactor-test initiate -action fill-in-missing-abstract-methods -in=%s:33:1-end -in=%s:34:1-end %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

struct FinalTarget final : AbstractClass {
};
// CHECK2: Initiated the 'fill-in-missing-abstract-methods' action at [[@LINE-2]]:1

// RUN: clang-refactor-test initiate -action fill-in-missing-abstract-methods -in=%s:38:1-end -in=%s:39:1-2 %s | FileCheck --check-prefix=CHECK2 %s

union Union { };
// RUN: not clang-refactor-test initiate -action fill-in-missing-abstract-methods -in=%s:44:1-end %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

class NoAbstractParents : Base { };
// RUN: not clang-refactor-test initiate -action fill-in-missing-abstract-methods -in=%s:47:1-35 %s 2>&1 | FileCheck --check-prefix=CHECK-NO-ABSTRACT-BASE %s
// CHECK-NO-ABSTRACT-BASE: Failed to initiate the refactoring action (The class has no abstract bases)

// RUN: not clang-refactor-test initiate -action fill-in-missing-abstract-methods -in=%s:1:1-end -in=%s:6:1-end %s 2>&1 | FileCheck --check-prefix=CHECK-NO-ABSTRACT-BASE %s

// Check selection:

// RUN: clang-refactor-test initiate -action fill-in-missing-abstract-methods -selected=%s:10:1-21:2 -selected=%s:12:1-16:10 -selected=%s:11:1-20:2 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: not clang-refactor-test initiate -action fill-in-missing-abstract-methods -selected=%s:13:3-14:4 -selected=%s:16:3-16:27 -selected=%s:18:3-20:4 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

struct HasAllMethods: AbstractClass {
   virtual void method() override { }
};
// RUN: not clang-refactor-test initiate -action fill-in-missing-abstract-methods -at=%s:58:1 %s 2>&1 | FileCheck --check-prefix=CHECK-NO-MISSING-METHODS %s
// CHECK-NO-MISSING-METHODS: Failed to initiate the refactoring action (The class has no missing abstract class methods)

// Shouldn't crash:
// forward-decl: +1:1
struct ForwardDecl;

// RUN: not clang-refactor-test initiate -action fill-in-missing-abstract-methods -at=forward-decl %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
