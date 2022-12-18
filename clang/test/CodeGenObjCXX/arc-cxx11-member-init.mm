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

// CHECK: [[mClipData:%.*]] = getelementptr inbounds %class.XClipboardDataSet, ptr
// CHECK: [[CLS:%.*]] = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_"
// CHECK: [[SEL:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
// CHECK: [[CALL:%.*]] = call noundef ptr @objc_msgSend(ptr noundef [[CLS]], ptr noundef [[SEL]])
// CHECK: store ptr [[CALL]], ptr [[mClipData]], align 8

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
// CHECK-NOT: define internal noundef ptr @"\01-[Foo .cxx_construct
