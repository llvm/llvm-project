
@interface EmptySelectorsRule_Psych

- (int):(int)_; // CHECK1: rename [[@LINE]]:8 -> [[@LINE]]:8
- (void)test: (int)x :(int)y; // CHECK2: rename [[@LINE]]:9 -> [[@LINE]]:13, [[@LINE]]:22 -> [[@LINE]]:22
- (void):(int)_ :(int) m:(int)z; // CHECK3: rename [[@LINE]]:9 -> [[@LINE]]:9, [[@LINE]]:17 -> [[@LINE]]:17, [[@LINE]]:25 -> [[@LINE]]:25

@end

namespace g {
    int x;
}

@implementation EmptySelectorsRule_Psych

- (int):(int)_ { // CHECK1: rename [[@LINE]]:8 -> [[@LINE]]:8
    [self :2];   // CHECK1: rename [[@LINE]]:11 -> [[@LINE]]:11
    SEL s0 = @selector(:); // CHECK1: selector [[@LINE]]:24 -> [[@LINE]]:24
    // CHECK1-NOT: comment
    // CHECK1-NOT: rename
    // CHECK1-NOT: selector
// RUN: clang-refactor-test rename-indexed-file -name=: -new-name=foo -indexed-file=%s -indexed-symbol-kind=objc-im -indexed-at=4:8 -indexed-at=16:8 -indexed-at=objc-message:17:11 %s | FileCheck --check-prefix=CHECK1 %s
    return 0;
}
- (void)test: (int)x :(int)y { } // CHECK2: rename [[@LINE]]:9 -> [[@LINE]]:13, [[@LINE]]:22 -> [[@LINE]]:22
- (void) :(int)_ :(int)m :(int)z { // CHECK3: rename [[@LINE]]:10 -> [[@LINE]]:10, [[@LINE]]:18 -> [[@LINE]]:18, [[@LINE]]:26 -> [[@LINE]]:26
    [self test:0:1]; // CHECK2: rename [[@LINE]]:11 -> [[@LINE]]:15, [[@LINE]]:17 -> [[@LINE]]:17
    SEL s1 = @selector(test::); // CHECK2: selector [[@LINE]]:24 -> [[@LINE]]:28, [[@LINE]]:29 -> [[@LINE]]:29
    @selector(test: :); // CHECK2: selector [[@LINE]]:15 -> [[@LINE]]:19, [[@LINE]]:21 -> [[@LINE]]:21
    // CHECK2-NOT: comment
    // CHECK2-NOT: rename
    // CHECK2-NOT: selector
// RUN: clang-refactor-test rename-indexed-file -name=test:: -new-name=foo:bar: -indexed-file=%s -indexed-symbol-kind=objc-im -indexed-at=5:9 -indexed-at=25:9 -indexed-at=objc-message:27:11 %s | FileCheck --check-prefix=CHECK2 %s

    [self: ::g::x + ([self: 0]):~0 :3]; // CHECK3: rename [[@LINE]]:10 -> [[@LINE]]:10, [[@LINE]]:32 -> [[@LINE]]:32, [[@LINE]]:36 -> [[@LINE]]:36
    SEL s2 = @selector(:::); // CHECK3: selector [[@LINE]]:24 -> [[@LINE]]:24, [[@LINE]]:25 -> [[@LINE]]:25, [[@LINE]]:26 -> [[@LINE]]:26
    @selector(: ::); // CHECK3: selector [[@LINE]]:15 -> [[@LINE]]:15, [[@LINE]]:17 -> [[@LINE]]:17, [[@LINE]]:18 -> [[@LINE]]:18
    @selector(::::); // not matching.
    // CHECK3-NOT: comment
    // CHECK3-NOT: rename
    // CHECK3-NOT: selector
// RUN: clang-refactor-test rename-indexed-file -name=::: -new-name=do:stuff:: -indexed-file=%s -indexed-symbol-kind=objc-im -indexed-at=6:9 -indexed-at=26:10 -indexed-at=objc-message:35:10 %s | FileCheck --check-prefix=CHECK3 %s

    // NO Textual matches: text:
    // :
    // :::
}

@end
