// RUN: %clang -fexceptions -S -emit-llvm %s -o -

@interface B 
-(int)bar;
@end

@interface A
-(void) Foo:(int) state;
@end

@implementation A 
- (void) Foo:(int) state {

        int wasResponded = 0;
        @try {
        if (state) {
           B * b = 0;
           @try { }
           @finally {
             wasResponded = ![b bar];
           }
        }
        }
        @finally {
        }
}
@end
