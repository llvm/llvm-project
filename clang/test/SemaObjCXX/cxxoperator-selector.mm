// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// rdar:// 8328250

@class NSDate;

@interface XSGraphDataSet 
- and;
- xor;
- or;

- ||; // expected-error {{expected selector for Objective-C method}}

- &&; // expected-error {{expected selector for Objective-C method}}

- (void)dataSetForValuesBetween:(NSDate *)startDate and:(NSDate *)endDate;
@end

@implementation XSGraphDataSet
- (id) and{return 0; };
- (id) xor{return 0; };
- (id) or{return 0; };

- (void)dataSetForValuesBetween:(NSDate *)startDate and:(NSDate *)endDate { return; }
@end
