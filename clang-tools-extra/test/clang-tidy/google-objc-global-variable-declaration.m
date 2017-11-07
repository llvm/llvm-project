// RUN: %check_clang_tidy %s google-objc-global-variable-declaration %t

@class NSString;
static NSString* const myConstString = @"hello";
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: const global variable 'myConstString' must have a name which starts with 'k[A-Z]' [google-objc-global-variable-declaration]
// CHECK-FIXES: static NSString* const kMyConstString = @"hello";

static NSString* MyString = @"hi";
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: non-const global variable 'MyString' must have a name which starts with 'g[A-Z]' [google-objc-global-variable-declaration]
// CHECK-FIXES: static NSString* gMyString = @"hi";

NSString* globalString = @"test";
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: non-const global variable 'globalString' must have a name which starts with 'g[A-Z]' [google-objc-global-variable-declaration]
// CHECK-FIXES: NSString* gGlobalString = @"test";

static NSString* a = @"too simple";
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: non-const global variable 'a' must have a name which starts with 'g[A-Z]' [google-objc-global-variable-declaration]
// CHECK-FIXES: static NSString* a = @"too simple";

static NSString* noDef;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: non-const global variable 'noDef' must have a name which starts with 'g[A-Z]' [google-objc-global-variable-declaration]
// CHECK-FIXES: static NSString* gNoDef;

static NSString* const _notAlpha = @"NotBeginWithAlpha";
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: const global variable '_notAlpha' must have a name which starts with 'k[A-Z]' [google-objc-global-variable-declaration]
// CHECK-FIXES: static NSString* const _notAlpha = @"NotBeginWithAlpha";

static NSString* const k_Alpha = @"SecondNotAlpha";
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: const global variable 'k_Alpha' must have a name which starts with 'k[A-Z]' [google-objc-global-variable-declaration]
// CHECK-FIXES: static NSString* const k_Alpha = @"SecondNotAlpha";

static NSString* const kGood = @"hello";
// CHECK-MESSAGES-NOT: :[[@LINE-1]]:24: warning: const global variable 'kGood' must have a name which starts with 'k[A-Z]' [google-objc-global-variable-declaration]
static NSString* gMyIntGood = 0;
// CHECK-MESSAGES-NOT: :[[@LINE-1]]:18: warning: non-const global variable 'gMyIntGood' must have a name which starts with 'g[A-Z]' [google-objc-global-variable-declaration]
@implementation Foo
- (void)f {
    int x = 0;
// CHECK-MESSAGES-NOT: :[[@LINE-1]]:9: warning: non-const global variable 'gMyIntGood' must have a name which starts with 'g[A-Z]' [google-objc-global-variable-declaration]
}
@end
