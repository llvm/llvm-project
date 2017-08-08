// RUN: %clang_analyze_cc1  -analyzer-checker=alpha.security.taint,debug.TaintTest %s -verify
// expected-no-diagnostics

#import <stdarg.h>

@interface NSString
- (NSString *)stringByAppendingString:(NSString *)aString;
@end
extern void NSLog (NSString *format, ...);
extern void NSLogv(NSString *format, va_list args);

void TestLog (NSString *format, ...);
void TestLog (NSString *format, ...) {
    va_list ap;
    va_start(ap, format);
    NSString *string = @"AAA: ";
    
    NSLogv([string stringByAppendingString:format], ap);
    
    va_end(ap);
}