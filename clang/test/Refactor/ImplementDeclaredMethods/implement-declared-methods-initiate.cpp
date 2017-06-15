
struct Class {
  int field;

  Class();

  Class(int x) { }

  ~Class();

  // commment
  void method();

  virtual voidMethod(int y) const;
  void implementedMethod() const {

  }

  void outOfLineImpl(int x);

  void anotherImplementedMethod() {

  }
};
// CHECK1: Initiated the 'implement-declared-methods' action at [[@LINE-20]]:3
// CHECK2: Initiated the 'implement-declared-methods' action at [[@LINE-17]]:3
// CHECK3: Initiated the 'implement-declared-methods' action at [[@LINE-15]]:3
// CHECK4: Initiated the 'implement-declared-methods' action at [[@LINE-14]]:3

void function();

void function() {

}

void Class::outOfLineImpl(int x) {

}

// RUN: clang-refactor-test initiate -action implement-declared-methods -in=%s:5:3-10 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test initiate -action implement-declared-methods -in=%s:9:3-11 %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test initiate -action implement-declared-methods -in=%s:12:3-16 %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test initiate -action implement-declared-methods -in=%s:14:3-34 %s | FileCheck --check-prefix=CHECK4 %s

// RUN: not clang-refactor-test initiate -action implement-declared-methods -in=%s:2:1-end -in=%s:3:1-end -in=%s:4:1-end -in=%s:5:1-2 -in=%s:7:1-end -in=%s:9:1-2 -in=%s:11:1-end -in=%s:15:1-end -in=%s:16:1-end -in=%s:17:1-end -in=%s:19:1-end %s -in=%s:30:1-end 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// CHECK-NO: Failed to initiate the refactoring action

// RUN: clang-refactor-test list-actions -at=%s:5:3 %s | FileCheck --check-prefix=CHECK-ACTION %s
// CHECK-ACTION: Generate Missing Function Definitions

// Class, ~Class, method, voidMethod:
// CHECK5: Initiated the 'implement-declared-methods' action at [[@LINE-48]]:3 -> [[@LINE-39]]:34
// RUN: clang-refactor-test initiate -action implement-declared-methods -selected=%s:2:1-24:3 -selected=%s:3:1-23:1 -selected=%s:4:1-23:3 -selected=%s:5:1-14:35 -selected=%s:5:9-14:4  %s | FileCheck --check-prefix=CHECK5 %s

// ~Class, method
// CHECK6: Initiated the 'implement-declared-methods' action at [[@LINE-48]]:3 -> [[@LINE-45]]:16
// RUN: clang-refactor-test initiate -action implement-declared-methods -selected=%s:9:1-12:16 -selected=%s:7:1-13:1 -selected=%s:7:17-12:4  %s | FileCheck --check-prefix=CHECK6 %s

// voidMethod
// CHECK7: Initiated the 'implement-declared-methods' action at [[@LINE-47]]:3
// RUN: clang-refactor-test initiate -action implement-declared-methods -selected=%s:14:3-14:10 -selected=%s:14:22-14:27 %s | FileCheck --check-prefix=CHECK7 %s

// RUN: not clang-refactor-test initiate -action implement-declared-methods -selected=%s:2:1-30:10 -selected=%s:15:1-15:10 -selected=%s:16:1-16:3 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
