// RUN: %clang_cc1 -triple arm64e-apple-ios13.0 -debug-info-kind=standalone -fobjc-arc -fsanitize=nullability-return \
// RUN:   %s -emit-llvm -o - | FileCheck %s

@interface NSObject
+ (id)alloc;
@end

@interface NSString : NSObject
@end

// CHECK: define {{.*}}@"\01-[MyData setData:]"
// CHECK: [[DATA:%.*]] = alloca %struct.Data
// CHECK: call %struct.Data* @_ZN4DataD1Ev(%struct.Data* [[DATA]]){{.*}}, !dbg [[DATA_PROPERTY_LOC:![0-9]+]]
// CHECK-NEXT: ret void

// CHECK: define {{.*}}@"\01-[MyData string]"
// CHECK: call void @__ubsan_handle_nullability_return_v1_abort{{.*}}, !dbg [[STRING_PROPERTY_LOC:![0-9]+]]
// CHECK: ret

@interface MyData : NSObject
struct Data {
    NSString *name;
};

// CHECK-DAG: [[DATA_PROPERTY_LOC]] = !DILocation(line: [[@LINE+1]]
@property struct Data data;

// CHECK-DAG: [[STRING_PROPERTY_LOC]] = !DILocation(line: [[@LINE+1]]
@property (nonnull) NSString *string;

@end

@implementation MyData
@end
