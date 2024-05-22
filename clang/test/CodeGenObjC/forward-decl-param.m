// RUN: %clang_cc1 %s -emit-llvm -o - 

// crash due to forward-declared struct in protocol method parameter.

@protocol P
- (void) A:(struct z) z;
@end
@interface I < P >
@end
@implementation I
@end

@interface I2
- (void) A:(struct z2) z2;
@end
@implementation I2
@end

