@interface I1

@property int p1;

@end

@implementation I1

- (void)foo {
  self.p1;         // CHECK: rename [[@LINE]]:8 -> [[@LINE]]:10
  [self p1];       // CHECK: rename [[@LINE]]:9 -> [[@LINE]]:11
  [self setP1: 2]; // CHECK: rename "setFoo" [[@LINE]]:9 -> [[@LINE]]:14
  _p1 = 3;         // CHECK: rename "_foo" [[@LINE]]:3 -> [[@LINE]]:6
}

@end

// RUN: clang-refactor-test rename-indexed-file -name=p1 -name=p1 -name=setP1 -name=_p1 -new-name=foo -new-name=foo -new-name=setFoo -new-name=_foo -indexed-file=%s -indexed-at=10:8 -indexed-at=objc-im:1:11:9 -indexed-at=objc-im:2:12:9 -indexed-at=3:13:3 %s | FileCheck %s

// p1 _p1 setP1
// CHECK: comment [[@LINE-1]]:4
// CHECK: comment "_foo" [[@LINE-2]]:7
// CHECK: comment "setFoo" [[@LINE-3]]:11

@selector(p1)
@selector(setP1:)
@selector(_p1)
// CHECK: selector [[@LINE-3]]:11
// CHECK: selector "setFoo" [[@LINE-3]]:11
// CHECK-NOT: selector

@interface ImplicitProperty

- (int)implicit; // IMPL_GET: rename [[@LINE]]:8 -> [[@LINE]]:16
- (void)setImplicit:(int)x; // IMPL_SET: rename [[@LINE]]:9 -> [[@LINE]]:20

@end

void useImplicitProperty(ImplicitProperty *x) {
  x.implicit; // IMPL_GET: rename [[@LINE]]:5 -> [[@LINE]]:13
  x.implicit = 0; // IMPL_SET: implicit-property [[@LINE]]:5 -> [[@LINE]]:13
}

// RUN: clang-refactor-test rename-indexed-file -name=implicit -new-name=foo -indexed-file=%s -indexed-at=objc-im:34:8 -indexed-at=objc-message:40:5 %s | FileCheck --check-prefix=IMPL_GET %s
// RUN: clang-refactor-test rename-indexed-file -name=setImplicit -new-name=setFoo -indexed-file=%s -indexed-at=objc-im:35:9 -indexed-at=objc-message:41:5 %s | FileCheck --check-prefix=IMPL_SET %s
