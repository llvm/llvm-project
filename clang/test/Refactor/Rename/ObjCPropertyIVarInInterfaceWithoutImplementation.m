@interface ExplicitIVarsInInterface {
  int _p1; // CHECK1: rename "_foo" [[@LINE]]:7 -> [[@LINE]]:10
  @public
  int _p2; // CHECK2: rename "_foo" [[@LINE]]:7 -> [[@LINE]]:10
}

@property int p1; // CHECK1: rename [[@LINE]]:15 -> [[@LINE]]:17
@property int p2; // CHECK2: rename [[@LINE]]:15 -> [[@LINE]]:17

@end

void explicitIVarsInInterface(ExplicitIVarsInInterface* object) {
  object->_p7 = // CHECK1: rename "_foo" [[@LINE]]:11 -> [[@LINE]]:14
                object->_p8; // CHECK2: rename "_foo" [[@LINE]]:25 -> [[@LINE]]:28
}

// XFAIL: *
// This test is currently disabled as renaming can't initiate a property
// renaming operation in a TU without @implementation.
// rdar://29329980

// RUN: clang-refactor-test rename-initiate -at=%s:2:7 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:7:15 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:13:11 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s

// RUN: clang-refactor-test rename-initiate -at=%s:4:7 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:8:15 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:14:25 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
