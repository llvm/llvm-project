// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -fblocks -verify -Wno-objc-root-class %s
// expected-no-diagnostics
// rdar://10156674

@class NSArray;

@interface MyClass2  {
@private
    NSArray *_names1;
    NSArray *_names2;
    NSArray *_names3;
    NSArray *_names4;
}
@property (readwrite, strong) NSArray *names1; // <-- warning: Type of property....
- (void)setNames1:(NSArray *)names;
@property (readwrite, strong) __strong NSArray *names2; // <-- warning: Type of property....
- (void)setNames2:(NSArray *)names;
@property (readwrite, strong) __strong NSArray *names3; // <-- OK
- (void)setNames3:(__strong NSArray *)names;
@property (readwrite, strong) NSArray *names4; // <-- warning: Type of property....
- (void)setNames4:(__strong NSArray *)names;

@end

@implementation MyClass2
- (NSArray *)names1 { return _names1; }
- (void)setNames1:(NSArray *)names {}
- (NSArray *)names2 { return _names2; }
- (void)setNames2:(NSArray *)names {}
- (NSArray *)names3 { return _names3; }
- (void)setNames3:(__strong NSArray *)names {}
- (NSArray *)names4 { return _names4; }
- (void)setNames4:(__strong NSArray *)names {}

@end

