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
// CHECK-LABEL: define dso_local noundef %struct.type_info* @"?test0_typeid@@YAPBUtype_info@@XZ"()
// CHECK:   ret %struct.type_info* bitcast (%rtti.TypeDescriptor2* @"??_R0H@8" to %struct.type_info*)

const std::type_info* test1_typeid() { return &typeid(A); }
// CHECK-LABEL: define dso_local noundef %struct.type_info* @"?test1_typeid@@YAPBUtype_info@@XZ"()
// CHECK:   ret %struct.type_info* bitcast (%rtti.TypeDescriptor7* @"??_R0?AUA@@@8" to %struct.type_info*)

const std::type_info* test2_typeid() { return &typeid(&a); }
// CHECK-LABEL: define dso_local noundef %struct.type_info* @"?test2_typeid@@YAPBUtype_info@@XZ"()
// CHECK:   ret %struct.type_info* bitcast (%rtti.TypeDescriptor7* @"??_R0PAUA@@@8" to %struct.type_info*)

const std::type_info* test3_typeid() { return &typeid(*fn()); }
// CHECK-LABEL: define dso_local noundef %struct.type_info* @"?test3_typeid@@YAPBUtype_info@@XZ"()
// CHECK:        [[CALL:%.*]] = call noundef %struct.A* @"?fn@@YAPAUA@@XZ"()
// CHECK-NEXT:   [[CMP:%.*]] = icmp eq %struct.A* [[CALL]], null
// CHECK-NEXT:   br i1 [[CMP]]
// CHECK:        call i8* @__RTtypeid(i8* null)
// CHECK-NEXT:   unreachable
// CHECK:        [[THIS:%.*]] = bitcast %struct.A* [[CALL]] to i8*
// CHECK-NEXT:   [[VBTBLP:%.*]] = getelementptr %struct.A, %struct.A* [[CALL]], i32 0, i32 0
// CHECK-NEXT:   [[VBTBL:%.*]] = load i32*, i32** [[VBTBLP]], align 4
// CHECK-NEXT:   [[VBSLOT:%.*]] = getelementptr inbounds i32, i32* [[VBTBL]], i32 1
// CHECK-NEXT:   [[VBASE_OFFS:%.*]] = load i32, i32* [[VBSLOT]], align 4
// CHECK-NEXT:   [[ADJ:%.*]] = getelementptr inbounds i8, i8* [[THIS]], i32 [[VBASE_OFFS]]
// CHECK-NEXT:   [[RT:%.*]] = call i8* @__RTtypeid(i8* nonnull [[ADJ]])
// CHECK-NEXT:   [[RET:%.*]] = bitcast i8* [[RT]] to %struct.type_info*
// CHECK-NEXT:   ret %struct.type_info* [[RET]]

const std::type_info* test4_typeid() { return &typeid(b); }
// CHECK: define dso_local noundef %struct.type_info* @"?test4_typeid@@YAPBUtype_info@@XZ"()
// CHECK:   ret %struct.type_info* bitcast (%rtti.TypeDescriptor2* @"??_R0H@8" to %struct.type_info*)

const std::type_info* test5_typeid() { return &typeid(v); }
// CHECK: define dso_local noundef %struct.type_info* @"?test5_typeid@@YAPBUtype_info@@XZ"()
// CHECK:   ret %struct.type_info* bitcast (%rtti.TypeDescriptor7* @"??_R0?AUV@@@8" to %struct.type_info*)

const std::type_info *test6_typeid() { return &typeid((V &)v); }
// CHECK: define dso_local noundef %struct.type_info* @"?test6_typeid@@YAPBUtype_info@@XZ"()
// CHECK:   ret %struct.type_info* bitcast (%rtti.TypeDescriptor7* @"??_R0?AUV@@@8" to %struct.type_info*)

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
// CHECK: %[[cp:.*]] = catchpad within %[[cs]] [i8* null, i32 64, i8* null]
// CHECK: invoke i8* @__RTtypeid(i8* {{.*}}) [ "funclet"(token %[[cp]]) ]
}
