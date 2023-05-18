typedef const void *CFTypeRef;

void test(id x) {
  (__bridge CFTypeRef)x;
}



// RUN: c-index-test -code-completion-at=%s:4:4 %s -fobjc-arc -fobjc-nonfragile-abi | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: macro definition:{TypedText __autoreleasing} (70)
// CHECK-CC1: NotImplemented:{TypedText __bridge}{HorizontalSpace  }{Placeholder type}{RightParen )}{Placeholder expression} (40)
// CHECK-CC1: NotImplemented:{TypedText __bridge_retained}{HorizontalSpace  }{Placeholder CF type}{RightParen )}{Placeholder expression} (40)
// CHECK-CC1: NotImplemented:{TypedText __bridge_transfer}{HorizontalSpace  }{Placeholder Objective-C type}{RightParen )}{Placeholder expression} (40)
// CHECK-CC1: macro definition:{TypedText __strong} (70)
// CHECK-CC1: macro definition:{TypedText __unsafe_unretained} (70)
// CHECK-CC1: macro definition:{TypedText __weak} (70)
