// RUN: %clang_cc1  -triple x86_64-apple-darwin10 -fobjc-arc -std=c++11 -emit-llvm -o - %s | FileCheck %s
// rdar://16299964
  
@interface NSObject
+ (id)new;
@end

@interface NSMutableDictionary : NSObject
@end
  
class XClipboardDataSet
{ 
  NSMutableDictionary* mClipData = [NSMutableDictionary new];
};
  
@interface AppDelegate @end

@implementation AppDelegate
- (void)applicationDidFinishLaunching
{ 
 XClipboardDataSet clip; 
}
@end

// CHECK: [[mClipData:%.*]] = getelementptr inbounds %class.XClipboardDataSet, %class.XClipboardDataSet*
// CHECK: [[CLS:%.*]] = load %struct._class_t*, %struct._class_t** @"OBJC_CLASSLIST_REFERENCES_$_"
// CHECK: [[RECV:%.*]] = bitcast %struct._class_t* [[CLS]] to i8*
// CHECK: [[SEL:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
// CHECK: [[CALL:%.*]] = call noundef i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* noundef [[RECV]], i8* noundef [[SEL]])
// CHECK: [[THREE:%.*]] = bitcast i8* [[CALL]] to [[T:%.*]]*
// CHECK: store [[T]]* [[THREE]], [[T]]** [[mClipData]], align 8

// rdar://18950072
struct Butt { };

__attribute__((objc_root_class))
@interface Foo {
  Butt x;
  Butt y;
  Butt z;
}
@end
@implementation Foo
@end
// CHECK-NOT: define internal noundef i8* @"\01-[Foo .cxx_construct
