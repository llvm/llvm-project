// RUN: %check_clang_tidy %s objc-property-declaration %t \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: objc-property-declaration.Acronyms, value: "ABC;TGIF"}]}' \
// RUN: --
@class NSString;

@interface Foo
@property(assign, nonatomic) int AbcNotRealPrefix;
// CHECK-MESSAGES: :[[@LINE-1]]:34: warning: property name 'AbcNotRealPrefix' not using lowerCamelCase style or not prefixed in a category, according to the Apple Coding Guidelines [objc-property-declaration]
// CHECK-FIXES: @property(assign, nonatomic) int abcNotRealPrefix;
@property(assign, nonatomic) int ABCCustomPrefix;
@property(strong, nonatomic) NSString *ABC_custom_prefix;
// CHECK-MESSAGES: :[[@LINE-1]]:40: warning: property name 'ABC_custom_prefix' not using lowerCamelCase style or not prefixed in a category, according to the Apple Coding Guidelines [objc-property-declaration]
@property(assign, nonatomic) int GIFShouldIncludeStandardAcronym;
@end
