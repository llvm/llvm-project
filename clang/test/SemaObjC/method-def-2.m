// RUN: %clang_cc1 -ast-print %s
extern void abort(void);
#define CHECK_IF(expr) if(!(expr)) abort()

static double d = 4.5920234e2;

@interface Foo 
-(void) brokenType: (int)x floatingPoint: (double)y;
@end


@implementation Foo
-(void) brokenType: (int)x floatingPoint: (double)y
{
        CHECK_IF(x == 459);
        CHECK_IF(y == d);
}
@end

