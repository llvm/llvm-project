// RUN: %check_clang_tidy %s google-objc-global-variable-declaration %t

@class NSString;
static NSString* const myConstString = @"hello";
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: const global variable 'myConstString' must have a name which starts with an appropriate prefix [google-objc-global-variable-declaration]
// CHECK-FIXES: static NSString* const kMyConstString = @"hello";

class MyTest {
    static int not_objc_style;
};
