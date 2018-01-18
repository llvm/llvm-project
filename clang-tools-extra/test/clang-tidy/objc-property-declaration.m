// RUN: %check_clang_tidy %s objc-property-declaration %t
@class NSData;
@class NSString;
@class UIViewController;

@interface Foo
@property(assign, nonatomic) int NotCamelCase;
// CHECK-MESSAGES: :[[@LINE-1]]:34: warning: property name 'NotCamelCase' should use lowerCamelCase style, according to the Apple Coding Guidelines [objc-property-declaration]
// CHECK-FIXES: @property(assign, nonatomic) int notCamelCase;
@property(assign, nonatomic) int camelCase;
@property(strong, nonatomic) NSString *URLString;
@property(strong, nonatomic) NSString *bundleID;
@property(strong, nonatomic) NSData *RGBABytes;
@property(strong, nonatomic) UIViewController *notificationsVC;
@property(strong, nonatomic) NSString *URL_string;
// CHECK-MESSAGES: :[[@LINE-1]]:40: warning: property name 'URL_string' should use lowerCamelCase style, according to the Apple Coding Guidelines [objc-property-declaration]
@end
