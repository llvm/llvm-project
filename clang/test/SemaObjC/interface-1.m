// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 %s -fsyntax-only -verify
// rdar://5957506

@interface NSWhatever :
NSObject     // expected-error {{cannot find interface declaration for 'NSObject'}}
<NSCopying>  // expected-error {{cannot find protocol declaration for 'NSCopying'}}
@end


// rdar://6095245
@interface A
{
  int x
}  // expected-error {{expected ';' at end of declaration list}}
@end


// rdar://4304469
@interface INT1
@end

void test2() {
    // rdar://6827200
    INT1 b[3];          // expected-error {{array of interface 'INT1' is invalid (probably should be an array of pointers)}}
    INT1 *c = &b[0];
    ++c;
}


// rdar://6611778
@interface FOO  // expected-note {{previous definition is here}}
- (void)method;
@end

@interface FOO  // expected-error {{duplicate interface definition for class 'FOO'}}
- (void)method2;
@end

