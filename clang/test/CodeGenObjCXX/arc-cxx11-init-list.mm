// RUN: %clang_cc1 -no-enable-noundef-analysis -triple armv7-ios5.0 -std=c++11 -fmerge-all-constants -fobjc-arc -Os -emit-llvm -o - %s | FileCheck %s

// CHECK: @[[STR0:.*]] = private unnamed_addr constant [5 x i8] c"str0\00", section "__TEXT,__cstring,cstring_literals"
// CHECK: @[[UNNAMED_CFSTRING0:.*]] = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @[[STR0]], i32 4 }, section "__DATA,__cfstring"
// CHECK: @[[STR1:.*]] = private unnamed_addr constant [5 x i8] c"str1\00", section "__TEXT,__cstring,cstring_literals"
// CHECK: @[[UNNAMED_CFSTRING1:.*]] = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @[[STR1]], i32 4 }, section "__DATA,__cfstring"
// CHECK: @[[REFTMP:.*]] = private constant [2 x ptr] [ptr @[[UNNAMED_CFSTRING0]], ptr @[[UNNAMED_CFSTRING1]]]

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

// CHECK: [[INSTANCE:%.*]] = {{.*}} call ptr @objc_msgSend(ptr {{.*}}, ptr {{.*}})
// CHECK-NEXT: store ptr [[INSTANCE]], ptr %{{.*}},
// CHECK: call void @llvm.objc.release(ptr {{.*}})

extern "C" void multiple() { function({ [I new], [I new] }); }

// CHECK: [[INSTANCE:%.*]] = {{.*}} call ptr @objc_msgSend(ptr {{.*}}, ptr {{.*}})
// CHECK-NEXT: store ptr [[INSTANCE]], ptr %{{.*}},
// CHECK: call void @llvm.objc.release(ptr {{.*}})

std::initializer_list<id> foo1() {
  return {@"str0", @"str1"};
}

// CHECK: define{{.*}} void @_Z4foo1v(ptr {{.*}} %[[AGG_RESULT:.*]])
// CHECK: store ptr @[[REFTMP]], ptr %[[AGG_RESULT]]
// CHECK: %[[SIZE:.*]] = getelementptr inbounds %"class.std::initializer_list.0", ptr %[[AGG_RESULT]], i32 0, i32 1
// CHECK: store i32 2, ptr %[[SIZE]]
// CHECK: ret void

void external();

extern "C" void extended() {
  const auto && temporary = { [I new] };
  external();
}

// CHECK: [[INSTANCE:%.*]] = {{.*}} call ptr @objc_msgSend(ptr {{.*}}, ptr {{.*}})
// CHECK: {{.*}} call void @_Z8externalv()
// CHECK: {{.*}} call void @llvm.objc.release(ptr {{.*}})

std::initializer_list<I *> il = { [I new] };

// CHECK: [[POOL:%.*]] = {{.*}} call ptr @llvm.objc.autoreleasePoolPush()
// CHECK: [[INSTANCE:%.*]] = {{.*}} call ptr @objc_msgSend(ptr {{.*}}, ptr {{.*}})
// CHECK-NEXT: store ptr [[INSTANCE]], ptr @_ZGR2il_
// CHECK: {{.*}} call void @llvm.objc.autoreleasePoolPop(ptr [[POOL]])
