// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 %s -fsyntax-only -verify

@interface NSWhatever :
NSObject     // expected-error {{cannot find interface declaration for 'NSObject'}}
<NSCopying>  // expected-error {{no type or protocol named 'NSCopying'}}
@end

@interface A
{
  int x
}  // expected-error {{expected ';' at end of declaration list}}
@end

@interface INT1
@end

void test2(void) {
    INT1 b[3];          // expected-error {{array of interface 'INT1' is invalid (probably should be an array of pointers)}}
    INT1 *c = &b[0];
    ++c;
}

@interface FOO  // expected-note {{previous definition is here}}
- (void)method;
@end

@interface FOO  // expected-error {{duplicate interface definition for class 'FOO'}}
- (void)method2;
@end

