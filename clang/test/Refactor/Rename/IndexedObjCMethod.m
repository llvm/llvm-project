@interface Test

- (int)performAction:(int)action with:(int)value; // CHECK: rename [[@LINE]]:8 -> [[@LINE]]:21, [[@LINE]]:34 -> [[@LINE]]:38

@end

@implementation Test

- (int)performAction:(int)action
       with:(int)value { // CHECK: rename [[@LINE-1]]:8 -> [[@LINE-1]]:21, [[@LINE]]:8 -> [[@LINE]]:12
    return action + value;
}

+ (void)foo:(Test*)t {
    [t performAction: 2 with: 4]; // CHECK: rename [[@LINE]]:8 -> [[@LINE]]:21, [[@LINE]]:25 -> [[@LINE]]:29
    SEL s1 = @selector(performAction:
                       with:);    // CHECK-NOT: selector [[@LINE-1]]:24 -> [[@LINE-1]]:37, [[@LINE]]:24 -> [[@LINE]]:28
    SEL s2 = @selector(performAction:); // CHECK-NOT: selector [[@LINE]]
    SEL s3 = @selector(performAction);  // CHECK-NOT: selector [[@LINE]]
    // Not indexed
    [t performAction: 1 with: 2]; // CHECK-NOT: rename [[@LINE]]
}

@end

// RUN: clang-refactor-test rename-indexed-file -no-textual-matches -name=performAction:with -new-name=foo:bar -indexed-file=%s -indexed-at=3:8 -indexed-at=9:8 -indexed-at=objc-message:15:8 -indexed-symbol-kind=objc-im %s | FileCheck %s

@interface SemicolonIsExcluded
-(void)/*A_propA4_set_decl*/setPropA4X:(int)value;
@end
// CHECK2: rename [[@LINE-2]]:29 -> [[@LINE-2]]:39

// RUN: clang-refactor-test rename-indexed-file -no-textual-matches -name=setPropA4X: -new-name=foo -indexed-file=%s -indexed-at=29:29 -indexed-symbol-kind=objc-im %s -x objective-c | FileCheck --check-prefix=CHECK2 %s

// It should be possible to have the filename as one of the compilation arguments
// RUN: clang-refactor-test rename-indexed-file -no-textual-matches -ignore-filename-for-initiation-tu -name=performAction:with: -new-name=foo:bar: -indexed-file=%s -indexed-at=3:8 -indexed-at=9:8 -indexed-at=objc-message:15:8 -indexed-symbol-kind=objc-cm %s -c %s -Wall | FileCheck %s
// RUN: clang-refactor-test rename-indexed-file -no-textual-matches -ignore-filename-for-initiation-tu -name=performAction:with: -new-name=foo:bar: -indexed-file=%s -indexed-at=3:8 -indexed-at=9:8 -indexed-at=objc-message:15:8 -indexed-symbol-kind=objc-cm %s %s -fsyntax-only | FileCheck %s

// -gmodules should be stripped to avoid -fmodule-format=obj in CC1 arguments:
// RUN: clang-refactor-test rename-indexed-file -no-textual-matches -name=performAction:with: -new-name=foo:bar: -indexed-file=%s -indexed-at=3:8 -indexed-at=9:8 -indexed-at=objc-message:15:8 -indexed-symbol-kind=objc-cm %s -fmodules -gmodules | FileCheck %s

// These texual matches should be reported as comment or selector occurrences:
// CHECK3: rename [[@LINE-40]]:8 -> [[@LINE-40]]:21, [[@LINE-40]]:34 -> [[@LINE-40]]:38
// performAction
/* performAction: with: 2 performAction */
/*! performAction+1
// performAction with
!*/
/// Hello performAction with World
/// \c performAction.

// CHECK3: comment [[@LINE-8]]:4 -> [[@LINE-8]]:17
// CHECK3-NEXT: comment [[@LINE-8]]:4 -> [[@LINE-8]]:17
// CHECK3-NEXT: comment [[@LINE-9]]:27 -> [[@LINE-9]]:40
// CHECK3-NEXT: documentation [[@LINE-9]]:5 -> [[@LINE-9]]:18
// CHECK3-NEXT: documentation [[@LINE-9]]:4 -> [[@LINE-9]]:17
// CHECK3-NEXT: documentation [[@LINE-8]]:11 -> [[@LINE-8]]:24
// CHECK3-NEXT: documentation [[@LINE-8]]:8 -> [[@LINE-8]]:21

// "performAction:with:"
// 'performAction:'with:
// CHECK3-NEXT: comment [[@LINE-2]]:5 -> [[@LINE-2]]:18
// CHECK3-NEXT: comment [[@LINE-2]]:5 -> [[@LINE-2]]:18

// CHECK3-NEXT: selector [[@LINE+1]]:11 -> [[@LINE+1]]:24, [[@LINE+1]]:25 -> [[@LINE+1]]:29
@selector(performAction:with:);
// CHECK3-NEXT: selector [[@LINE+1]]:11 -> [[@LINE+1]]:24, [[@LINE+1]]:28 -> [[@LINE+1]]:32
@selector(performAction :  with );
// CHECK3-NEXT: selector [[@LINE+2]]:19 -> [[@LINE+2]]:32, [[@LINE+2]]:46 -> [[@LINE+2]]:50
SEL s = @selector(//comment
                  performAction: /*comment*/ with
                  );
// CHECK3-NEXT: selector [[@LINE+1]]:33 -> [[@LINE+1]]:46, [[@LINE+2]]:33 -> [[@LINE+2]]:37
void call = @selector(@selector(performAction:
                                with: ));

// CHECK3-NEXT: comment [[@LINE+1]]:55
// RUN: clang-refactor-test rename-indexed-file -name=performAction:with: -new-name=foo:bar -indexed-file=%s -indexed-at=objc-cm:3:8 %s | FileCheck --check-prefix=CHECK3 %s

// These ones shouldn't:
// performAction2 PERFORMACTION performActionWith
const char *test = "performAction:with:";

@selector(performAction: with ::)
@selector(performAction:)
@selector(performAction)
@selector(performAction with)
@selector(performAction:without:)
@selector(performAction:with:somethingElse:)
@selector(performAction:with "")
@selector("performAction:with:")
@selector(with: performAction:)
selector(performAction:with)
(performAction:with:)

// CHECK3-NOT: comment
// CHECK3-NOT: documentation
// CHECK3-NOT: selector
// CHECK3-NOT: string-literal
// It should be possible to find a selector in a file without any indexed occurrences:

// CHECK4: selector [[@LINE+1]]:11
@selector(nonIndexedSelector)
// CHECK4-NEXT: comment
// CHECK4-NOT: selector

// RUN: clang-refactor-test rename-indexed-file -indexed-symbol-kind=objc-im -name=nonIndexedSelector -new-name=test -indexed-file=%s %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test rename-indexed-file -indexed-symbol-kind=objc-cm -name=nonIndexedSelector -new-name=test -indexed-file=%s %s | FileCheck --check-prefix=CHECK4 %s

#define MACRO doStuff
#define MACRO2(x, y) doStuff:(x) with: (y)

@interface I
- (void)doStuff:(int)x with: y;
@end

@implementation I
- (void)MACRO:(int)x with: y {
  [self MACRO2(x, y)];
}
@end

// CHECK-MACRO: macro [[@LINE-5]]:9 -> [[@LINE-5]]:9
// CHECK-MACRO-NEXT: macro [[@LINE-5]]:9 -> [[@LINE-5]]:9
// CHECK-MACRO-NEXT: rename [[@LINE-11]]:9 -> [[@LINE-11]]:16, [[@LINE-11]]:24 -> [[@LINE-11]]:28
// CHECK-MACRO-NOT: macro

// RUN: clang-refactor-test rename-indexed-file -indexed-symbol-kind=objc-im -name=doStuff:with: -new-name=foo:bar -indexed-file=%s -indexed-at=114:9 -indexed-at=118:9 -indexed-at=119:9 %s | FileCheck --check-prefix=CHECK-MACRO %s
