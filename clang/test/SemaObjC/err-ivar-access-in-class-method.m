// RUN: %clang_cc1 -x objective-c -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://10593227

@class UIWindow;

@interface CNAppDelegate

@property (strong, nonatomic) UIWindow *window;

@end


@interface CNAppDelegate ()
@property (nonatomic,retain) id foo;
@end

@implementation CNAppDelegate
@synthesize foo;
@synthesize window = _window;

+(void)myClassMethod;
{
        foo = 0; // expected-error {{instance variable 'foo' accessed in class method}}
}
@end
