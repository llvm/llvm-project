// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o -

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
