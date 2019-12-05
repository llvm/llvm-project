// RUN: %clang_cc1 -triple arm64e-apple-ios13.0 -debug-info-kind=standalone -fobjc-arc \
// RUN:   %s -emit-llvm -o - | FileCheck %s

@interface NSObject
+ (id)alloc;
@end

@interface NSString : NSObject
@end

// CHECK: define {{.*}}@"\01-[MyData setData:]"
// CHECK: [[DATA:%.*]] = alloca %struct.Data
// CHECK: call %struct.Data* @_ZN4DataD1Ev(%struct.Data* [[DATA]]){{.*}}, !dbg [[LOC:![0-9]+]]
// CHECK-NEXT: ret void

// [[LOC]] = !DILocation(line: 0

@interface MyData : NSObject
struct Data {
    NSString *name;
};
@property struct Data data;
@end
@implementation MyData
@end
