enum Color {
  Black,
  Blue,
  White,
  Gold
};

void initiate(Color c, int i) {
  switch (c) {
  case Black:
    break;
  }
// CHECK1: Initiated the 'fill-in-enum-switch-cases' action at [[@LINE-4]]:3

  switch (c) {
  }
// CHECK2: Initiated the 'fill-in-enum-switch-cases' action at [[@LINE-2]]:3
}

// RUN: clang-refactor-test list-actions -at=%s:9:3 %s | FileCheck --check-prefix=CHECK-ACTION %s
// CHECK-ACTION: Add Missing Switch Cases

// Ensure the the action can be initiated around a switch:

// RUN: clang-refactor-test initiate -action fill-in-enum-switch-cases -in=%s:9:3-15 -in=%s:10:1-14 -in=%s:11:1-11 -in=%s:12:1-3 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test initiate -action fill-in-enum-switch-cases -in=%s:15:3-15 -in=%s:16:1-4 %s | FileCheck --check-prefix=CHECK2 %s

// Ensure that the action can't be initiated in other places:

// RUN: not clang-refactor-test initiate -action fill-in-enum-switch-cases -in=%s:8:1-32 -in=%s:9:1-2 -in=%s:13:1-77 -in=%s:15:1-2 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
// CHECK-NO: Failed to initiate the refactoring action

void dontInitiate(Color c, int i) {
  switch (c) {
  case Black:
    break;
  case Blue:
    break;
  case White:
    break;
  case Gold:
    break;
  }

  switch (i) {
  case 0:
    break;
  }

  switch ((int)c) {
  case 0:
    break;
  }
}

// Ensure that the action can't be initiated on switches that have all cases or
// that don't work with an enum.

// RUN: not clang-refactor-test initiate -action fill-in-enum-switch-cases -in=%s:34:3-15 %s 2>&1 | FileCheck --check-prefix=CHECK-ALL-COVERED %s
// CHECK-ALL-COVERED: Failed to initiate the refactoring action (All enum cases are already covered)!
// RUN: not clang-refactor-test initiate -action fill-in-enum-switch-cases -in=%s:45:3-15 -in=%s:50:3-20 %s 2>&1 | FileCheck --check-prefix=CHECK-NOT-ENUM %s
// CHECK-NOT-ENUM: Failed to initiate the refactoring action (The switch doesn't operate on an enum)!

void initiateWithDefault(Color c, int i) {
  switch (c) {
  case Black:
    break;
  default:
    break;
  }
// CHECK3: Initiated the 'fill-in-enum-switch-cases' action at [[@LINE-6]]:3

  switch (c) {
  default:
    break;
  }
// CHECK4: Initiated the 'fill-in-enum-switch-cases' action at [[@LINE-4]]:3
}

// RUN: clang-refactor-test initiate -action fill-in-enum-switch-cases -at=%s:65:3 %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test initiate -action fill-in-enum-switch-cases -at=%s:73:3 %s | FileCheck --check-prefix=CHECK4 %s

enum class Shape {
  Rectangle,
  Circle,
  Octagon
};

typedef enum {
  Anon1,
  Anon2
} AnonymousEnum;

void initiateEnumClass(Shape shape, AnonymousEnum anon) {
  switch (shape) {
  }
// CHECK5: Initiated the 'fill-in-enum-switch-cases' action at [[@LINE-2]]:3
  switch (anon) {
  }
// CHECK6: Initiated the 'fill-in-enum-switch-cases' action at [[@LINE-2]]:3
}

// RUN: clang-refactor-test initiate -action fill-in-enum-switch-cases -at=%s:95:3 %s -std=c++11 | FileCheck --check-prefix=CHECK5 %s
// RUN: clang-refactor-test initiate -action fill-in-enum-switch-cases -at=%s:98:3 %s | FileCheck --check-prefix=CHECK6 %s

// Ensure that the operation can be initiated from a selection:

// RUN: clang-refactor-test initiate -action fill-in-enum-switch-cases -selected=%s:9:3-12:4 -selected=%s:9:15-12:3 -selected=%s:9:15-12:3 -selected=%s:10:3-11:10  %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test initiate -action fill-in-enum-switch-cases -selected=%s:15:3-16:4  %s | FileCheck --check-prefix=CHECK2 %s

void dontInitiateSelectedBody(Shape shape) {
  switch (shape) {
  case Shape::Rectangle: {
    break;
  }
  case Shape::Circle:
    break;
  }
}

// RUN: not clang-refactor-test initiate -action fill-in-enum-switch-cases -selected=%s:114:5-114:11 -selected=%s:117:5-117:10 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

enum IncompleteEnum : int;
enum class IncompleteClassEnum : short;
enum class IncompleteClassEnum2;
void dontInitiateOnIncompleteEnum(IncompleteEnum e1, IncompleteClassEnum e2, IncompleteClassEnum2 e3) {
  switch (e1) {
  }
  switch (e1) {
  case 0:
    break;
  }
  switch (e2) {
  }
  switch (e2) {
  case (IncompleteClassEnum)0:
    break;
  }
  switch (e3) {
  }
}

// RUN: not clang-refactor-test initiate -action fill-in-enum-switch-cases -at=%s:127:3 -at=%s:129:3 -at=%s:133:3 -at=%s:135:3 -at=%s:139:3 %s -std=c++11 2>&1 | FileCheck --check-prefix=CHECK-NOT-COMPLETE %s
// CHECK-NOT-COMPLETE: Failed to initiate the refactoring action (The enum type is incomplete)!

void initiateWhenSelectionIsPartial() {
  int partiallySelected = 0;
}
int global = 0;
// RUN: not clang-refactor-test initiate -action fill-in-enum-switch-cases -selected=%s:147:1-150:1 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
