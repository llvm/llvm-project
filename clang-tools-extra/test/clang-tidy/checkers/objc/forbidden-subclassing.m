// RUN: %check_clang_tidy %s objc-forbidden-subclassing %t

@interface UIImagePickerController
@end

@interface Foo : UIImagePickerController
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: Objective-C interface 'Foo' subclasses 'UIImagePickerController', which is not intended to be subclassed [objc-forbidden-subclassing]
@end

// Check subclasses of subclasses.
@interface Bar : Foo
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: Objective-C interface 'Bar' subclasses 'UIImagePickerController', which is not intended to be subclassed [objc-forbidden-subclassing]
@end

@interface Baz
@end

// Make sure innocent subclasses aren't caught by the check.
@interface Blech : Baz
// CHECK-MESSAGES-NOT: :[[@LINE-1]]:12: warning: Objective-C interface 'Blech' subclasses 'Baz', which is not intended to be subclassed [objc-forbidden-subclassing]
@end
