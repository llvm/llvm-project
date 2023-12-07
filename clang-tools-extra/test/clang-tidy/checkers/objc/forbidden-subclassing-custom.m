// RUN: %check_clang_tidy %s objc-forbidden-subclassing %t \
// RUN: -config='{CheckOptions: \
// RUN:  {objc-forbidden-subclassing.ClassNames: "Foo;Quux"}}' \
// RUN: --

@interface UIImagePickerController
@end

// Make sure custom config options replace (not add to) the default list.
@interface Waldo : UIImagePickerController
// CHECK-MESSAGES-NOT: :[[@LINE-1]]:12: warning: Objective-C interface 'Waldo' subclasses 'UIImagePickerController', which is not intended to be subclassed [objc-forbidden-subclassing]
@end

@interface Foo
@end

@interface Bar : Foo
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: Objective-C interface 'Bar' subclasses 'Foo', which is not intended to be subclassed [objc-forbidden-subclassing]
@end

// Check subclasses of subclasses.
@interface Baz : Bar
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: Objective-C interface 'Baz' subclasses 'Foo', which is not intended to be subclassed [objc-forbidden-subclassing]
@end

@interface Quux
@end

// Check that more than one forbidden superclass can be specified.
@interface Xyzzy : Quux
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: Objective-C interface 'Xyzzy' subclasses 'Quux', which is not intended to be subclassed [objc-forbidden-subclassing]
@end

@interface Plugh
@end

@interface Corge : Plugh
// CHECK-MESSAGES-NOT: :[[@LINE-1]]:12: warning: Objective-C interface 'Corge' subclasses 'Plugh', which is not intended to be subclassed [objc-forbidden-subclassing]
@end
