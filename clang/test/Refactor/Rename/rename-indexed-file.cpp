// Note: the run lines follow their respective tests, since line/column
// matter in this test

class Test {  // CHECK1: rename [[@LINE]]:7 -> [[@LINE]]:11
public:
  Test() { }  // CHECK1: rename [[@LINE]]:3 -> [[@LINE]]:7
  ~Test() { } // CHECK1: rename [[@LINE]]:4 -> [[@LINE]]:8

  void doSomething() { return; }
  void otherFile();
};

void foo() {
  Test test;  // CHECK1: rename [[@LINE]]:3 -> [[@LINE]]:7
  (test).doSomething();
}

Test notIndexed; // CHECK1-NOT: rename [[@LINE]]

// RUN: clang-refactor-test rename-indexed-file -name=Test -new-name=Foo -indexed-file=%s -indexed-at=4:7 -indexed-at=6:3 -indexed-at=7:4 -indexed-at=14:3 %s | FileCheck --check-prefix=CHECK1 %s

// RUN: clang-refactor-test rename-indexed-file -no-textual-matches -name=Test -new-name=Foo -indexed-file=%S/Inputs/rename-indexed-file.cpp -indexed-at=1:6 -indexed-at=2:3 -indexed-at=3:6 %s | FileCheck --check-prefix=CHECK2 %s
// CHECK2: rename 1:6 -> 1:10
// CHECK2: rename 2:3 -> 2:7
// CHECK2: rename 3:6 -> 3:10

// A valid location with an non-identifier token shouldn't produce an occurence
// RUN: clang-refactor-test rename-indexed-file -no-textual-matches -name=Test -new-name=Foo -indexed-file=%s -indexed-at=15:3 %s | FileCheck --check-prefix=CHECK3 %s

// A invalid location shouldn't produce an occurence
// RUN: clang-refactor-test rename-indexed-file -no-textual-matches -name=Test -new-name=Foo -indexed-file=%s -indexed-at=999:1 %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-indexed-file -no-textual-matches -name=Test -new-name=Foo -indexed-file=%s -indexed-at=0:1 %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-indexed-file -no-textual-matches -name=Test -new-name=Foo -indexed-file=%s -indexed-at=1:0 %s | FileCheck --check-prefix=CHECK3 %s


// CHECK3: no replacements found
// CHECK3-NOT: rename

// RUN: not clang-refactor-test rename-indexed-file -no-textual-matches -name=Test -new-name=Foo %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR1 %s

// CHECK-ERROR1: for the -indexed-file option: must be specified at least once!

// It should be possible to have the filename as one of the compilation arguments
// RUN: clang-refactor-test rename-indexed-file -no-textual-matches -ignore-filename-for-initiation-tu -name=Test -new-name=Foo -indexed-file=%s -indexed-at=4:7 -indexed-at=6:3 -indexed-at=7:4 -indexed-at=14:3 %s -c %s -Wall | FileCheck --check-prefix=CHECK1 %s

// -gmodules should be stripped to avoid -fmodule-format=obj in CC1 arguments:
// RUN: clang-refactor-test rename-indexed-file -no-textual-matches -name=Test -new-name=Foo -indexed-file=%s -indexed-at=4:7 -indexed-at=6:3 -indexed-at=7:4 -indexed-at=14:3 %s -fmodules -gmodules | FileCheck --check-prefix=CHECK1 %s

// These texual matches should be reported as comment occurrences:
// CHECK4-INIT: rename [[@LINE-46]]:7 -> [[@LINE-46]]:11
// Test
/* Test 2 Test */
/** Test+1
// Test
**/
/// Hello Test World
//! \c Test.

// CHECK4: comment [[@LINE-8]]:4 -> [[@LINE-8]]:8
// CHECK4-NEXT: comment [[@LINE-8]]:4 -> [[@LINE-8]]:8
// CHECK4-NEXT: comment [[@LINE-9]]:11 -> [[@LINE-9]]:15
// CHECK4-NEXT: documentation [[@LINE-9]]:5 -> [[@LINE-9]]:9
// CHECK4-NEXT: documentation [[@LINE-9]]:4 -> [[@LINE-9]]:8
// CHECK4-NEXT: documentation [[@LINE-8]]:11 -> [[@LINE-8]]:15
// CHECK4-NEXT: documentation [[@LINE-8]]:8 -> [[@LINE-8]]:12

// "Test"
// 'Test'
// CHECK4-NEXT: comment [[@LINE-2]]:5 -> [[@LINE-2]]:9
// CHECK4-NEXT: comment [[@LINE-2]]:5 -> [[@LINE-2]]:9

// CHECK4-NEXT: comment [[@LINE+1]]:55
// RUN: clang-refactor-test rename-indexed-file -name=Test -new-name=Foo -indexed-file=%s -indexed-at=4:7 %s | FileCheck --check-prefixes=CHECK4-INIT,CHECK4 %s
// We should find textual occurrences even without indexed occurrences:
// CHECK4-NEXT: comment [[@LINE+1]]:55
// RUN: clang-refactor-test rename-indexed-file -name=Test -new-name=Foo -indexed-file=%s %s | FileCheck --check-prefix=CHECK4 %s

// These ones shouldn't:
// Test2 test Testable
/// _Test
/// ATest_
const char *test = "Test";
void Test20() { }

// CHECK4-NOT: comment
// CHECK4-NOT: documentation


class MyInclude { // CHECK5: rename [[@LINE]]:7 -> [[@LINE]]:16
};

 /*comment*/ #include "MyInclude.h"
#include <clang/myinclude.h>
#import <MyInclude/ThisIsMyInclude>
// CHECK5-NEXT: filename [[@LINE-3]]:24 -> [[@LINE-3]]:33
// CHECK5-NEXT: filename [[@LINE-3]]:17 -> [[@LINE-3]]:26
// CHECK5-NEXT: filename [[@LINE-3]]:26 -> [[@LINE-3]]:35

// CHECK5-NOT: filename
#include "My Include.h"
"MyInclude.h"

// RUN: clang-refactor-test rename-indexed-file -name=MyInclude -new-name=Foo -indexed-file=%s -indexed-at=89:7 -indexed-at=include:92:1 -indexed-at=include:93:1 -indexed-at=include:94:1 -indexed-at=include:100:1 %s | FileCheck --check-prefix=CHECK5 %s

#define MACRO variable

void macroOccurrence() {
  variable;
  MACRO;
  22;
  MACRO;
}
// CHECK-MACRO: rename [[@LINE-5]]:3 -> [[@LINE-5]]:11
// CHECK-MACRO-NEXT: macro [[@LINE-5]]:3 -> [[@LINE-5]]:3
// CHECK-MACRO-NOT: macro

// RUN: clang-refactor-test rename-indexed-file -name=variable -new-name=foo -indexed-file=%s -indexed-at=108:3 -indexed-at=109:3 -indexed-at=110:3 -indexed-at=111:2 %s | FileCheck --check-prefix=CHECK-MACRO %s

struct MyType { // CHECK-MACRO-PREFIX: rename [[@LINE]]:8 -> [[@LINE]]:14
};
MyType MyTypePrefix; // CHECK-MACRO-PREFIX: macro [[@LINE]]:8 -> [[@LINE]]:8

// RUN: clang-refactor-test rename-indexed-file -name=MyType -new-name=x -indexed-file=%s -indexed-at=119:8 -indexed-at=121:8 %s | FileCheck --check-prefix=CHECK-MACRO-PREFIX %s
