// XFAIL: *
// TODO: Remove if unused
// Note: the run lines follow their respective tests, since line/column
// matter in this test

int variable = 0; // CHECK1: rename [[@LINE]]:5 -> [[@LINE]]:13

// RUN: clang-refactor-test rename-indexed-file -name=variable -new-name=class -indexed-file=%s -indexed-at=4:5 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: not clang-refactor-test rename-indexed-file -name=variable -new-name=class -indexed-file=%s -indexed-at=4:5 %s -x objective-c++ 2>&1 | FileCheck --check-prefix=CHECK-ERR %s
// CHECK-ERR: error: invalid new name

// RUN: not clang-refactor-test rename-indexed-file -name=variable -new-name=int -indexed-file=%s -indexed-at=4:5 %s 2>&1 | FileCheck --check-prefix=CHECK-ERR %s

@interface I

- (void)some:(int)x selector:(int)y; // CHECK2: rename [[@LINE]]:9 -> [[@LINE]]:13, [[@LINE]]:21 -> [[@LINE]]:29

@end

// RUN: clang-refactor-test rename-indexed-file -name=some:selector -new-name=struct:void -indexed-file=%s -indexed-at=14:9 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-indexed-file -name=some:selector: -new-name=struct:void -indexed-file=%s -indexed-at=14:9 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK2 %s

// RUN: not clang-refactor-test rename-indexed-file -name=some:selector -new-name=struct:void -indexed-file=%s -indexed-at=14:9 %s 2>&1 | FileCheck --check-prefix=CHECK-ERR-FAIL %s
// CHECK-ERR-FAIL: failed to perform indexed file rename
// RUN: not clang-refactor-test rename-indexed-file -name=some:selector -new-name=hello:123 -indexed-file=%s -indexed-at=14:9 -indexed-symbol-kind=objc-im %s 2>&1 | FileCheck --check-prefix=CHECK-ERR %s
// RUN: not clang-refactor-test rename-indexed-file -name=some:selector -new-name=+:test -indexed-file=%s -indexed-at=14:9 -indexed-symbol-kind=objc-im %s 2>&1 | FileCheck --check-prefix=CHECK-ERR %s

// RUN: not clang-refactor-test rename-indexed-file -name=some:selector -new-name=justOnePiece -indexed-file=%s -indexed-at=14:9 -indexed-symbol-kind=objc-im %s 2>&1 | FileCheck --check-prefix=CHECK-ERR2 %s
// CHECK-ERR2: error: the number of strings in the new name 'justOnePiece' doesn't match the the number of strings in the old name

@interface I1

- (void)singlePiece;

@end

// RUN: not clang-refactor-test rename-indexed-file -name=singlePiece -new-name=struct -indexed-file=%s -indexed-at=31:9 %s 2>&1 | FileCheck --check-prefix=CHECK-ERR %s
