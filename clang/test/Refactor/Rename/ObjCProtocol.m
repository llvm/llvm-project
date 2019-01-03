@protocol P1, // CHECK1: rename [[@LINE]]:11 -> [[@LINE]]:13
          P2; // CHECK2: rename [[@LINE]]:11 -> [[@LINE]]:13

@protocol P1 // CHECK1: rename [[@LINE]]:11 -> [[@LINE]]:13
@end

// RUN: clang-refactor-test rename-initiate -at=%s:1:11 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:4:11 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:2:11 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s

void protocolExpressions(id foo) {
  (void)@protocol(P1);    // CHECK1: rename [[@LINE]]:19 -> [[@LINE]]:21
  [foo p: @protocol(P2)]; // CHECK2: rename [[@LINE]]:21 -> [[@LINE]]:23
}

// RUN: clang-refactor-test rename-initiate -at=%s:12:19 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:13:21 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s

void qualifiedId(id<P1> foo) { // CHECK1: rename [[@LINE]]:21 -> [[@LINE]]:23
  id<P2, P1> bar = // CHECK2: rename [[@LINE]]:6 -> [[@LINE]]:8
                   // CHECK1: rename [[@LINE-1]]:10 -> [[@LINE-1]]:12
                   (id<P1, P2, P1>)foo; // CHECK1: rename [[@LINE]]:24 -> [[@LINE]]:26
                                        // CHECK2: rename [[@LINE-1]]:28 -> [[@LINE-1]]:30
                                        // CHECK1: rename [[@LINE-2]]:32 -> [[@LINE-2]]:34
}

// RUN: clang-refactor-test rename-initiate -at=%s:19:21 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:20:6 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:20:10 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:22:24 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:22:28 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:22:32 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s

typedef id<P1> TypedefQualifiedID; // CHECK1: rename [[@LINE]]:12 -> [[@LINE]]:14

// RUN: clang-refactor-test rename-initiate -at=%s:34:12 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s

@interface I1<P1> // CHECK1: rename [[@LINE]]:15 -> [[@LINE]]:17
@end

@protocol P3 < P1> // CHECK3: rename [[@LINE]]:11 -> [[@LINE]]:13
                   // CHECK1: rename [[@LINE-1]]:16 -> [[@LINE-1]]:18
@end

@interface I1 (Cat) <P2, P3> // CHECK2: rename [[@LINE]]:22 -> [[@LINE]]:24
@end                         // CHECK3: rename [[@LINE-1]]:26 -> [[@LINE-1]]:28

// RUN: clang-refactor-test rename-initiate -at=%s:38:15 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:41:16 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:45:22 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:41:11 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:45:26 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s

typedef I1<P1, P2> * TypedefI1; // CHECK1: rename [[@LINE]]:12 -> [[@LINE]]:14
                                // CHECK2: rename [[@LINE-1]]:16 -> [[@LINE-1]]:18

void qualifiedClassPointer(I1<P1> *x) { // CHECK1: rename [[@LINE]]:31 -> [[@LINE]]:33
}

// RUN: clang-refactor-test rename-initiate -at=%s:54:12 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:57:31 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:54:16 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s

void protocolTypeof(typeof(@protocol(P1)) *bar) { // CHECK1: rename [[@LINE]]:38 -> [[@LINE]]:40
}

// RUN: clang-refactor-test rename-initiate -at=%s:64:38 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
