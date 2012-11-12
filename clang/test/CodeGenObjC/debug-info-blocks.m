// REQUIRES: x86-64-registered-target
// RUN: %clang_cc1 -masm-verbose -S -fblocks -g  -triple x86_64-apple-darwin10 -fobjc-dispatch-method=mixed  %s -o - | FileCheck %s

//Radar 9279956
//CHECK:	## DW_OP_deref
//CHECK-NEXT:	## DW_OP_plus_uconst

typedef unsigned int NSUInteger;

@protocol NSObject
@end  
   
@interface NSObject <NSObject>
- (id)init;
+ (id)alloc;
@end 

@interface NSDictionary : NSObject 
- (NSUInteger)count;
@end    

@interface NSMutableDictionary : NSDictionary  
@end       

@interface A : NSObject {
@public
    int ivar;
}
@end

static void run(void (^block)(void))
{
    block();
}

@implementation A

- (id)init
{
    if ((self = [super init])) {
      run(^{
          NSMutableDictionary *d = [[NSMutableDictionary alloc] init]; 
          ivar = 42 + (int)[d count];
        });
    }
    return self;
}

@end

int main()
{
	A *a = [[A alloc] init];
	return 0;
}
