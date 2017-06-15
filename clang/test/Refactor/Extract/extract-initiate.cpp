struct Rectangle { int width, height; };

int sumArea(Rectangle *rs, int count) {
  int sum = 0;
  for (int i = 0; i < count; ++i) {
    Rectangle r = rs[i];
    sum += r.width * r.height;
  }
  return sum;
}

// RUN: clang-refactor-test list-actions -at=%s:7:30 -selected=%s:7:12-7:30 %s | FileCheck --check-prefix=CHECK-ACTION %s
// CHECK-ACTION: Extract Function

// Ensure the an entire expression can be extracted:

// RUN: clang-refactor-test initiate -action extract -selected=%s:7:12-7:30 %s | FileCheck --check-prefix=CHECK1 %s
// CHECK1: Initiated the 'extract' action at 7:12 -> 7:30

// Ensure that an expression can be extracted even when it's not fully selected:

// RUN: clang-refactor-test initiate -action extract -selected=%s:7:13-7:30 -selected=%s:7:18-7:30 -selected=%s:7:20-7:30 -selected=%s:7:19-7:30 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test initiate -action extract -selected=%s:7:12-7:29 -selected=%s:7:12-7:23 -selected=%s:7:12-7:21 -selected=%s:7:12-7:22 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test initiate -action extract -selected=%s:7:19-7:22 -selected=%s:7:20-7:21 -selected=%s:7:15-7:25 %s | FileCheck --check-prefix=CHECK1 %s

// Ensure that the action isn't allowed be when no expression is selected:

// RUN: not clang-refactor-test initiate -action extract -selected=%s:1:1-1:5 -selected=%s:2:1-2:1 -selected=%s:3:1-3:38 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
// RUN: not clang-refactor-test initiate -action extract -at=%s:1:1 -in=%s:3:1-38  %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
// CHECK-NO: Failed to initiate the refactoring action
// CHECK-NO-NOT: Initiated the 'extract' action

int multipleCandidates(Rectangle &r1, Rectangle &r2) {
  int x = r1.width + r1.height; // CHECK2-SINGLE: Initiated the 'extract' action at [[@LINE]]:11 -> [[@LINE]]:31
  int y = r1.width - r2.width * r2.height;
}

// RUN: clang-refactor-test initiate -action extract -selected=%s:34:20-34:31 -selected=%s:34:19-34:23 %s | FileCheck --check-prefix=CHECK2 %s
// CHECK2: Initiated the 'extract' action with multiple candidates:
// CHECK2-NEXT: + r1.height
// CHECK2-NEXT: r1.width + r1.height
// RUN: clang-refactor-test initiate -action extract -selected=%s:34:20-34:21 %s | FileCheck --check-prefix=CHECK2-SINGLE %s

// RUN: clang-refactor-test initiate -action extract -selected=%s:35:19-35:42 %s | FileCheck --check-prefix=CHECK3 %s
// CHECK3: Initiated the 'extract' action with multiple candidates:
// CHECK3-NEXT: - r2.width * r2.height
// CHECK3-NEXT: r1.width - r2.width * r2.height

void trimWhitespaceAndSemiColons(const Rectangle &r) {
  int x =  r.width   +   r.width * r.height; ;
  //CHECK4: Initiated the 'extract' action at [[@LINE-1]]:26 -> [[@LINE-1]]:44
  //CHECK5: Initiated the 'extract' action at [[@LINE-2]]:12 -> [[@LINE-2]]:44
}

// RUN: clang-refactor-test initiate -action extract -selected=%s:50:23-50:46 -selected=%s:50:23-50:45 %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test initiate -action extract -selected=%s:50:10-51:3 %s | FileCheck --check-prefix=CHECK5 %s

void disallowBlankStatements() {
  // comment
  ;

}

// RUN: not clang-refactor-test initiate -action extract -selected=%s:59:1-59:3 -selected=%s:60:1-62:1 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

void allowCommentSelection(int y) {
  char c = "//fake comment" [y] ;
}

// RUN: not clang-refactor-test initiate -action extract -selected=%s:67:13-67:16 %s 2>&1 | FileCheck --check-prefix=CHECK-SIMPLE %s
// CHECK-SIMPLE: Failed to initiate the refactoring action (the selected expression is too simple)

void extractStatements(const Rectangle &r) {
  (void)r;
  int area = r.width * r.height;
  int perimeter = r.width + r.height;
  int sum = area + perimeter;
  // CHECK6: Initiated the 'extract' action at [[@LINE-3]]:3 -> [[@LINE-2]]:38
  // CHECK7: Initiated the 'extract' action at [[@LINE-5]]:3 -> [[@LINE-2]]:30
}

// RUN: clang-refactor-test initiate -action extract -selected=%s:75:1-76:38 -selected=%s:74:11-77:3 -selected=%s:75:14-76:26 %s | FileCheck --check-prefix=CHECK6 %s

// RUN: clang-refactor-test initiate -action extract -selected=%s:74:1-77:30 -selected=%s:73:45-78:1 %s | FileCheck --check-prefix=CHECK7 %s

void extractStatementsCompoundChild(const Rectangle &r) {
  int x = 0;
  {
    int area = r.width * r.height;
  }
  int y = 0;
  int z = 1;
  // CHECK8: Initiated the 'extract' action at [[@LINE-5]]:3 -> [[@LINE-2]]:13
}

// RUN: clang-refactor-test initiate -action extract -selected=%s:89:5-91:12 %s | FileCheck --check-prefix=CHECK8 %s

void extractStatementsTrimComments(const Rectangle &r) {
  int x = 0;

  // comment
  int area = r.width * r.height;

  // another comment
  int y = 0;

  // trailing comment
}
// CHECK9: Initiated the 'extract' action at [[@LINE-7]]:3 -> [[@LINE-7]]:33
// CHECK10: Initiated the 'extract' action at [[@LINE-5]]:3 -> [[@LINE-5]]:13

// RUN: clang-refactor-test initiate -action extract -selected=%s:100:1-102:32 -selected=%s:101:6-104:21 -selected=%s:100:1-105:3  %s | FileCheck --check-prefix=CHECK9 %s
// RUN: clang-refactor-test initiate -action extract -selected=%s:103:1-105:12 -selected=%s:104:6-107:22 -selected=%s:103:1-108:1  %s | FileCheck --check-prefix=CHECK10 %s

// RUN: not clang-refactor-test initiate -action extract -selected=%s:101:1-101:13 -selected=%s:106:1-107:22 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

void disallowEmptyCompoundStatement() {
  // comment
}

// RUN: not clang-refactor-test initiate -action extract -selected=%s:118:1-118:13 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

void trimLeadingCompoundStatementComment() {
  // comment

  int x = 0;
}

// RUN: not clang-refactor-test initiate -action extract -selected=%s:124:1-124:13 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

void extractEntireCompoundStatement() {
  {
    int x = 0;
    int y = 0;
  }
  // CHECK11: Initiated the 'extract' action at [[@LINE-4]]:3 -> [[@LINE-1]]:4
  // CHECK12: Initiated the 'extract' action at [[@LINE-4]]:5 -> [[@LINE-3]]:15
  // CHECK13: Initiated the 'extract' action at [[@LINE-5]]:5 -> [[@LINE-4]]:15
}

// RUN: clang-refactor-test initiate -action extract -selected=%s:132:3-135:4  %s | FileCheck --check-prefix=CHECK11 %s
// RUN: clang-refactor-test initiate -action extract -selected=%s:132:3-135:3  %s | FileCheck --check-prefix=CHECK12 %s
// RUN: clang-refactor-test initiate -action extract -selected=%s:132:4-135:4  %s | FileCheck --check-prefix=CHECK13 %s

void disallowExtractionWhenSelectionRangeIsOutsideFunction() {
  int x = 0;
  int x = 1;
}

// RUN: not clang-refactor-test initiate -action extract -selected=%s:143:1-147:12 -selected=%s:146:3-150:3 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
