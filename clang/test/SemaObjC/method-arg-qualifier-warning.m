// RUN: %clang_cc1  -fsyntax-only -verify %s

typedef signed char BOOL;

@interface NSString
- (BOOL)isEqualToString:(NSString *)aString; // expected-note 2{{passing argument to parameter 'aString' here}}
@end

static const NSString * Identifier1 =   @"Identifier1";
static NSString const * Identifier2 =   @"Identifier2";
static NSString * const Identifier3 =   @"Identifier3";

int main (void) {
        
    [@"Identifier1" isEqualToString:Identifier1]; // expected-warning {{sending 'const NSString *' to parameter of type 'NSString *' discards qualifiers}}
    [@"Identifier2" isEqualToString:Identifier2]; // expected-warning {{sending 'const NSString *' to parameter of type 'NSString *' discards qualifiers}}
    [@"Identifier3" isEqualToString:Identifier3];
    return 0;
}

