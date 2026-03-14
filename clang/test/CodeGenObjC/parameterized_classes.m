// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fblocks -fobjc-arc -emit-llvm -o - %s | FileCheck %s

// Parameterized classes have no effect on code generation; this test
// mainly verifies that CodeGen doesn't assert when substituted types
// in uses of methods don't line up exactly with the parameterized
// types in the method declarations due to type erasure. "Not crash"
// is the only interesting criteria here.

@protocol NSObject
@end

@protocol NSCopying
@end

__attribute__((objc_root_class))
@interface NSObject <NSObject>
@end

@interface NSString : NSObject <NSCopying>
@end

@interface NSMutableArray<T> : NSObject <NSCopying>
@property (copy,nonatomic) T firstObject;
- (void)addObject:(T)object;
- (void)sortWithFunction:(int (*)(T, T))function;
- (void)getObjects:(T __strong *)objects length:(unsigned*)length;
- (T)objectAtIndexedSubscript:(unsigned)index;
- (void)setObject:(T)object atIndexedSubscript:(unsigned)index;
@end

NSString *getFirstObjectProp(NSMutableArray<NSString *> *array) {
  return array.firstObject;
}

NSString *getFirstObjectMethod(NSMutableArray<NSString *> *array) {
  return [array firstObject];
}

void addObject(NSMutableArray<NSString *> *array, NSString *obj) {
  [array addObject: obj];
}

int compareStrings(NSString *x, NSString *y) { return 0; }
int compareBlocks(NSString * (^x)(NSString *),
                  NSString * (^y)(NSString *)) { return 0; }

void sortTest(NSMutableArray<NSString *> *array,
              NSMutableArray<NSString * (^)(NSString *)> *array2) {
  [array sortWithFunction: &compareStrings];
  [array2 sortWithFunction: &compareBlocks];
}

void getObjectsTest(NSMutableArray<NSString *> *array) {
  NSString * __strong *objects;
  unsigned length;
  [array getObjects: objects length: &length];
}

void printMe(NSString *name) { }

// CHECK-LABEL: define{{.*}} void @blockTest
void blockTest(NSMutableArray<void (^)(void)> *array, NSString *name) {
  // CHECK-NOT: ret void
  // CHECK: call ptr @llvm.objc.retainBlock
  [array addObject: ^ { printMe(name); }];
  // CHECK-NOT: ret void
  array[0] = ^ { printMe(name); };
  // CHECK: call ptr @llvm.objc.retainBlock
  // CHECK: ret void
}

// CHECK-LABEL: define internal void @"\01-[Derived setDest:]
// CHECK: %[[SELFADDR:.*]] = alloca ptr
// CHECK: %[[AADDR:a.addr]] = alloca ptr
// CHECK: %[[V2:.*]] = load ptr, ptr %[[AADDR]]
// CHECK: %[[V3:.*]] = load ptr, ptr %[[SELFADDR]]
// CHECK: %[[IVAR:.*]] = load i64, ptr @"OBJC_IVAR_$_Base._destination"
// CHECK: %[[ADDPTR:.*]] = getelementptr inbounds i8, ptr %[[V3]], i64 %[[IVAR]]
// CHECK: call void @llvm.objc.storeStrong(ptr %[[ADDPTR]], ptr %[[V2]])

@interface Base<DestType> : NSObject {
  DestType _destination;
}
@end

@interface Derived : Base<NSObject *>
- (void)setDest:(NSObject *)a;
@end

@implementation Derived
- (void)setDest:(NSObject *)a {
  _destination = a;
}
@end

// CHECK-LABEL: define internal void @"\01-[C0 foo1]"(
// CHECK: {{.*}} = alloca
// CHECK: {{.*}} = alloca
// CHECK: %[[D:.*]] = alloca ptr
// CHECK: %[[TEMP:.*]] = alloca ptr
// CHECK: %[[V4:.*]] = load ptr, ptr %[[D]]
// CHECK: store ptr %[[V4]], ptr %[[TEMP]]
// CHECK: call void @objc_msgSend(ptr noundef %{{.*}}, ptr noundef %{{.*}}, ptr noundef %[[TEMP]])

@interface P0<ObjectType> : NSObject
- (void)m0:(ObjectType *)first;
@end

@interface C0 : NSObject
-(void)foo1;
@end

@implementation C0 {
  P0<NSString *> *x;
}

-(void)foo1 {
  NSString *d;
  [x m0:&d];
}
@end
