// RUN: %clang_cc1 -triple armv7-ios5.0 -std=c++11 -fmerge-all-constants -fobjc-arc -Os -emit-llvm -o - %s | FileCheck %s

// CHECK: @[[STR0:.*]] = private unnamed_addr constant [5 x i8] c"str0\00", section "__TEXT,__cstring,cstring_literals"
// CHECK: @[[UNNAMED_CFSTRING0:.*]] = private global %struct.__NSConstantString_tag { i32* getelementptr inbounds ([0 x i32], [0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @[[STR0]], i32 0, i32 0), i32 4 }, section "__DATA,__cfstring"
// CHECK: @[[STR1:.*]] = private unnamed_addr constant [5 x i8] c"str1\00", section "__TEXT,__cstring,cstring_literals"
// CHECK: @[[UNNAMED_CFSTRING1:.*]] = private global %struct.__NSConstantString_tag { i32* getelementptr inbounds ([0 x i32], [0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @[[STR1]], i32 0, i32 0), i32 4 }, section "__DATA,__cfstring"
// CHECK: @[[REFTMP:.*]] = private constant [2 x i8*] [i8* bitcast (%struct.__NSConstantString_tag* @[[UNNAMED_CFSTRING0]] to i8*), i8* bitcast (%struct.__NSConstantString_tag* @[[UNNAMED_CFSTRING1]] to i8*)]

typedef __SIZE_TYPE__ size_t;

namespace std {
template <typename _Ep>
class initializer_list {
  const _Ep* __begin_;
  size_t __size_;

  initializer_list(const _Ep* __b, size_t __s);
};
}

@interface I
+ (instancetype) new;
@end

void function(std::initializer_list<I *>);

extern "C" void single() { function({ [I new] }); }

// CHECK: [[INSTANCE:%.*]] = {{.*}} call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* {{.*}}, i8* {{.*}})
// CHECK-NEXT: [[CAST:%.*]] = bitcast [{{[0-9]+}} x %0*]* %{{.*}} to i8**
// CHECK-NEXT: store i8* [[INSTANCE]], i8** [[CAST]],
// CHECK: call void @objc_release(i8* {{.*}})

extern "C" void multiple() { function({ [I new], [I new] }); }

// CHECK: [[INSTANCE:%.*]] = {{.*}} call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* {{.*}}, i8* {{.*}})
// CHECK-NEXT: [[CAST:%.*]] = bitcast [{{[0-9]+}} x %0*]* %{{.*}} to i8**
// CHECK-NEXT: store i8* [[INSTANCE]], i8** [[CAST]],
// CHECK: call void @objc_release(i8* {{.*}})

std::initializer_list<id> foo1() {
  return {@"str0", @"str1"};
}

// CHECK: define void @_Z4foo1v(%"class.std::initializer_list.0"* {{.*}} %[[AGG_RESULT:.*]])
// CHECK: %[[BEGIN:.*]] = getelementptr inbounds %"class.std::initializer_list.0", %"class.std::initializer_list.0"* %[[AGG_RESULT]], i32 0, i32 0
// CHECK: store i8** getelementptr inbounds ([2 x i8*], [2 x i8*]* @[[REFTMP]], i32 0, i32 0), i8*** %[[BEGIN]]
// CHECK: %[[SIZE:.*]] = getelementptr inbounds %"class.std::initializer_list.0", %"class.std::initializer_list.0"* %[[AGG_RESULT]], i32 0, i32 1
// CHECK: store i32 2, i32* %[[SIZE]]
// CHECK: ret void

void external();

extern "C" void extended() {
  const auto && temporary = { [I new] };
  external();
}

// CHECK: [[INSTANCE:%.*]] = {{.*}} call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* {{.*}}, i8* {{.*}})
// CHECK: {{.*}} call void @_Z8externalv()
// CHECK: {{.*}} call void @objc_release(i8* {{.*}})

std::initializer_list<I *> il = { [I new] };

// CHECK: [[POOL:%.*]] = {{.*}} call i8* @objc_autoreleasePoolPush()
// CHECK: [[INSTANCE:%.*]] = {{.*}} call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* {{.*}}, i8* {{.*}})
// CHECK-NEXT: store i8* [[INSTANCE]], i8** bitcast ([1 x %0*]* @_ZGR2il_ to i8**)
// CHECK: {{.*}} call void @objc_autoreleasePoolPop(i8* [[POOL]])
