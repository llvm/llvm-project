// RUN: %clang_cc1  -fsyntax-only -fobjc-arc -fobjc-runtime-has-weak -Wsuper-class-method-mismatch -verify %s
// expected-no-diagnostics

@class NSString;

@interface Super
@property (nonatomic) NSString *thingy;
@property () __weak id PROP;
@end

@interface Sub : Super
@end

@implementation Sub
- (void)setThingy:(NSString *)val
{
  [super setThingy:val];
}
@synthesize PROP;
@end
