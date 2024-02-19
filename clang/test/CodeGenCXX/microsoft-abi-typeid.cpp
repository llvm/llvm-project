// RUN: %clang_cc1 -emit-llvm -O1 -o - -triple=i386-pc-win32 %s -fexceptions -fcxx-exceptions | FileCheck %s

struct type_info;
namespace std { using ::type_info; }

struct V { virtual void f(); };
struct A : virtual V { A(); };

extern A a;
extern V v;
extern int b;
A* fn();

const std::type_info* test0_typeid() { return &typeid(int); }
// CHECK-LABEL: define dso_local noundef nonnull ptr @"?test0_typeid@@YAPBUtype_info@@XZ"()
// CHECK:   ret ptr @"??_R0H@8"

const std::type_info* test1_typeid() { return &typeid(A); }
// CHECK-LABEL: define dso_local noundef nonnull ptr @"?test1_typeid@@YAPBUtype_info@@XZ"()
// CHECK:   ret ptr @"??_R0?AUA@@@8"

const std::type_info* test2_typeid() { return &typeid(&a); }
// CHECK-LABEL: define dso_local noundef nonnull ptr @"?test2_typeid@@YAPBUtype_info@@XZ"()
// CHECK:   ret ptr @"??_R0PAUA@@@8"

const std::type_info* test3_typeid() { return &typeid(*fn()); }
// CHECK-LABEL: define dso_local noundef ptr @"?test3_typeid@@YAPBUtype_info@@XZ"()
// CHECK:        [[CALL:%.*]] = tail call noundef ptr @"?fn@@YAPAUA@@XZ"()
// CHECK-NEXT:   [[CMP:%.*]] = icmp eq ptr [[CALL]], null
// CHECK-NEXT:   br i1 [[CMP]]
// CHECK:        call ptr @__RTtypeid(ptr null)
// CHECK-NEXT:   unreachable
// CHECK:        [[VBTBL:%.*]] = load ptr, ptr [[CALL]], align 4
// CHECK-NEXT:   [[VBSLOT:%.*]] = getelementptr inbounds i8, ptr [[VBTBL]], i32 4
// CHECK-NEXT:   [[VBASE_OFFS:%.*]] = load i32, ptr [[VBSLOT]], align 4
// CHECK-NEXT:   [[ADJ:%.*]] = getelementptr inbounds i8, ptr [[CALL]], i32 [[VBASE_OFFS]]
// CHECK-NEXT:   [[RT:%.*]] = tail call ptr @__RTtypeid(ptr nonnull [[ADJ]])
// CHECK-NEXT:   ret ptr [[RT]]

const std::type_info* test4_typeid() { return &typeid(b); }
// CHECK: define dso_local noundef nonnull ptr @"?test4_typeid@@YAPBUtype_info@@XZ"()
// CHECK:   ret ptr @"??_R0H@8"

const std::type_info* test5_typeid() { return &typeid(v); }
// CHECK: define dso_local noundef nonnull ptr @"?test5_typeid@@YAPBUtype_info@@XZ"()
// CHECK:   ret ptr @"??_R0?AUV@@@8"

const std::type_info *test6_typeid() { return &typeid((V &)v); }
// CHECK: define dso_local noundef nonnull ptr @"?test6_typeid@@YAPBUtype_info@@XZ"()
// CHECK:   ret ptr @"??_R0?AUV@@@8"

namespace PR26329 {
struct Polymorphic {
  virtual ~Polymorphic();
};

void f(const Polymorphic &poly) {
  try {
    throw;
  } catch (...) {
    Polymorphic cleanup;
    typeid(poly);
  }
}
// CHECK-LABEL: define dso_local void @"?f@PR26329@@YAXABUPolymorphic@1@@Z"(
// CHECK: %[[cs:.*]] = catchswitch within none [label %{{.*}}] unwind to caller
// CHECK: %[[cp:.*]] = catchpad within %[[cs]] [ptr null, i32 64, ptr null]
// CHECK: invoke ptr @__RTtypeid(ptr {{.*}}) [ "funclet"(token %[[cp]]) ]
}
