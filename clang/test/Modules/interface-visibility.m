// RUN: %clang_cc1 -fmodules -fobjc-arc -x objective-c-module-map %s -fmodule-name=Foo -verify

module Foo {}

#pragma clang module contents
#pragma clang module begin Foo

// expected-no-diagnostics

#pragma clang module build Foundation
module Foundation {}
#pragma clang module contents
#pragma clang module begin Foundation
@interface NSIndexSet
@end
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import Foundation

@interface NSIndexSet (Testing)
- (int)foo;
@end

static inline int test(NSIndexSet *obj) {
  return [obj foo];
}

#pragma clang module end
