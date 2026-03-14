// RUN: %clang_cc1 -x objective-c -Wno-objc-root-class -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: FileCheck  --input-file=%t-rw.cpp %s

typedef struct MyWidget {
  int a;
} MyWidget;

MyWidget gWidget = { 17 };

@protocol MyProto
- (MyWidget *)widget;
@end

@interface Foo 
@end

@interface Bar: Foo <MyProto>
@end

@interface Container 
+ (MyWidget *)elementForView:(Foo *)view;
@end

@implementation Foo
@end

@implementation Bar
- (MyWidget *)widget {
  return &gWidget;
}
@end

@implementation Container
+ (MyWidget *)elementForView:(Foo *)view
{
  MyWidget *widget = (void*)0;
  if (@protocol(MyProto)) {
    widget = [(id <MyProto>)view widget];
  }
  return widget;
}
@end

int main(void) {
  id view;
  MyWidget *w = [Container elementForView: view];

  return 0;
}

@class NSObject;

@interface NSProtocolChecker
+ (id)protocolCheckerWithTarget:(NSObject *)anObject protocol:(Protocol *)aProtocol;
@end

@protocol NSConnectionVersionedProtocol
@end


@interface NSConnection @end

@implementation NSConnection
- (void) Meth {
  [NSProtocolChecker protocolCheckerWithTarget:0 protocol:@protocol(NSConnectionVersionedProtocol)];
}
@end

// CHECK: static struct _protocol_t *_OBJC_PROTOCOL_REFERENCE_$_NSConnectionVersionedProtocol = &_OBJC_PROTOCOL_NSConnectionVersionedProtocol
// CHECK: sel_registerName("protocolCheckerWithTarget:protocol:"), (NSObject *)0, (Protocol *)_OBJC_PROTOCOL_REFERENCE_$_NSConnectionVersionedProtocol
