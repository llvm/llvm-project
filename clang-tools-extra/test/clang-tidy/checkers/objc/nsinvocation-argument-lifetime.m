// RUN: %check_clang_tidy %s objc-nsinvocation-argument-lifetime %t

__attribute__((objc_root_class))
@interface NSObject
@end

@interface NSInvocation : NSObject
- (void)getArgument:(void *)Arg atIndex:(int)Index;
- (void)getReturnValue:(void *)ReturnValue;
@end

@interface OtherClass : NSObject
- (void)getArgument:(void *)Arg atIndex:(int)Index;
@end

struct Foo {
  __unsafe_unretained id Field1;
  id Field2;
  int IntField;
};

void foo(NSInvocation *Invocation) {
  __unsafe_unretained id Arg2;
  id Arg3;
  // CHECK-FIXES: __unsafe_unretained id Arg3;
  NSObject __strong *Arg4;
  // CHECK-FIXES: NSObject __unsafe_unretained *Arg4;
  __weak id Arg5;
  // CHECK-FIXES: __unsafe_unretained id Arg5;
  id ReturnValue;
  // CHECK-FIXES: __unsafe_unretained id ReturnValue;
  void (^BlockArg1)(void);
  // CHECK-FIXES: __unsafe_unretained void (^BlockArg1)(void);
  __unsafe_unretained void (^BlockArg2)(void);
  int IntVar;
  struct Foo Bar;

  [Invocation getArgument:&Arg2 atIndex:2];
  [Invocation getArgument:&IntVar atIndex:2];

  [Invocation getArgument:&Arg3 atIndex:3];
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: NSInvocation '-getArgument:atIndex:' should only pass pointers to objects with ownership __unsafe_unretained [objc-nsinvocation-argument-lifetime]

  [Invocation getArgument:&Arg4 atIndex:4];
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: NSInvocation '-getArgument:atIndex:' should only pass pointers to objects with ownership __unsafe_unretained [objc-nsinvocation-argument-lifetime]

  [Invocation getArgument:&Arg5 atIndex:5];
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: NSInvocation '-getArgument:atIndex:' should only pass pointers to objects with ownership __unsafe_unretained [objc-nsinvocation-argument-lifetime]

  [Invocation getArgument:&BlockArg1 atIndex:6];
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: NSInvocation '-getArgument:atIndex:' should only pass pointers to objects with ownership __unsafe_unretained [objc-nsinvocation-argument-lifetime]

  [Invocation getArgument:&BlockArg2 atIndex:6];

  [Invocation getReturnValue:&ReturnValue];
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: NSInvocation '-getReturnValue:' should only pass pointers to objects with ownership __unsafe_unretained [objc-nsinvocation-argument-lifetime]

  [Invocation getArgument:(void *)0 atIndex:6];

  [Invocation getArgument:&Bar.Field1 atIndex:2];
  [Invocation getArgument:&Bar.Field2 atIndex:2];
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: NSInvocation '-getArgument:atIndex:' should only pass pointers to objects with ownership __unsafe_unretained [objc-nsinvocation-argument-lifetime]
  [Invocation getArgument:&Bar.IntField atIndex:2];
}

void bar(OtherClass *OC) {
  id Arg;
  [OC getArgument:&Arg atIndex:2];
}

@interface TestClass : NSObject {
@public
  id Argument1;
  __unsafe_unretained id Argument2;
  struct Foo Bar;
  int IntIvar;
}
@end

@implementation TestClass

- (void)processInvocation:(NSInvocation *)Invocation {
  [Invocation getArgument:&Argument1 atIndex:2];
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: NSInvocation '-getArgument:atIndex:' should only pass pointers to objects with ownership __unsafe_unretained [objc-nsinvocation-argument-lifetime]
  [Invocation getArgument:&self->Argument1 atIndex:2];
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: NSInvocation '-getArgument:atIndex:' should only pass pointers to objects with ownership __unsafe_unretained [objc-nsinvocation-argument-lifetime]
  [Invocation getArgument:&Argument2 atIndex:2];
  [Invocation getArgument:&self->Argument2 atIndex:2];
  [Invocation getArgument:&self->IntIvar atIndex:2];

  [Invocation getReturnValue:&(self->Bar.Field1)];
  [Invocation getReturnValue:&(self->Bar.Field2)];
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: NSInvocation '-getReturnValue:' should only pass pointers to objects with ownership __unsafe_unretained [objc-nsinvocation-argument-lifetime]
  [Invocation getReturnValue:&(self->Bar.IntField)];
}

@end

void baz(NSInvocation *Invocation, TestClass *Obj) {
  [Invocation getArgument:&Obj->Argument1 atIndex:2];
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: NSInvocation '-getArgument:atIndex:' should only pass pointers to objects with ownership __unsafe_unretained [objc-nsinvocation-argument-lifetime]
  [Invocation getArgument:&Obj->Argument2 atIndex:2];
}
