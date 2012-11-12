// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c %s 2>&1 | FileCheck %s
// rdar://11376372

@class NSObject;

@interface TestObj {
}
-(void)aMethodWithArg1:(NSObject*)arg1 arg2:(NSObject*)arg2;
@end

int main(int argc, char *argv[])
{
    TestObj *obj;
    [obj aMethodWithArg1:@"Arg 1 Good", arg2:@"Arg 2 Good"]; 
}

// CHECK: {14:39-14:40}:""
