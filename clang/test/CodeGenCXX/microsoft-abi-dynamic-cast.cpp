// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -O1 -o - -fexceptions -triple=i386-pc-win32 %s | FileCheck %s

struct S { char a; };
struct V { virtual void f(); };
struct A : virtual V {};
struct B : S, virtual V {};
struct T {};

T* test0() { return dynamic_cast<T*>((B*)0); }
// CHECK-LABEL: define dso_local noalias noundef %struct.T* @"?test0@@YAPAUT@@XZ"()
// CHECK:   ret %struct.T* null

T* test1(V* x) { return &dynamic_cast<T&>(*x); }
// CHECK-LABEL: define dso_local noundef %struct.T* @"?test1@@YAPAUT@@PAUV@@@Z"(%struct.V* noundef %x)
// CHECK:        [[CAST:%.*]] = bitcast %struct.V* %x to i8*
// CHECK-NEXT:   [[CALL:%.*]] = tail call i8* @__RTDynamicCast(i8* [[CAST]], i32 0, i8* bitcast (%rtti.TypeDescriptor7* @"??_R0?AUV@@@8" to i8*), i8* bitcast (%rtti.TypeDescriptor7* @"??_R0?AUT@@@8" to i8*), i32 1)
// CHECK-NEXT:   [[RET:%.*]] = bitcast i8* [[CALL]] to %struct.T*
// CHECK-NEXT:   ret %struct.T* [[RET]]

T* test2(A* x) { return &dynamic_cast<T&>(*x); }
// CHECK-LABEL: define dso_local noundef %struct.T* @"?test2@@YAPAUT@@PAUA@@@Z"(%struct.A* noundef %x)
// CHECK:        [[CAST:%.*]] = bitcast %struct.A* %x to i8*
// CHECK-NEXT:   [[VBPTRPTR:%.*]] = getelementptr %struct.A, %struct.A* %x, i32 0, i32 0
// CHECK-NEXT:   [[VBTBL:%.*]] = load i32*, i32** [[VBPTRPTR]], align 4
// CHECK-NEXT:   [[VBOFFP:%.*]] = getelementptr inbounds i32, i32* [[VBTBL]], i32 1
// CHECK-NEXT:   [[VBOFFS:%.*]] = load i32, i32* [[VBOFFP]], align 4
// CHECK-NEXT:   [[ADJ:%.*]] = getelementptr inbounds i8, i8* [[CAST]], i32 [[VBOFFS]]
// CHECK-NEXT:   [[CALL:%.*]] = tail call i8* @__RTDynamicCast(i8* [[ADJ]], i32 [[VBOFFS]], i8* bitcast (%rtti.TypeDescriptor7* @"??_R0?AUA@@@8" to i8*), i8* bitcast (%rtti.TypeDescriptor7* @"??_R0?AUT@@@8" to i8*), i32 1)
// CHECK-NEXT:   [[RET:%.*]] = bitcast i8* [[CALL]] to %struct.T*
// CHECK-NEXT:   ret %struct.T* [[RET]]

T* test3(B* x) { return &dynamic_cast<T&>(*x); }
// CHECK-LABEL: define dso_local noundef %struct.T* @"?test3@@YAPAUT@@PAUB@@@Z"(%struct.B* noundef %x)
// CHECK:        [[VOIDP:%.*]] = getelementptr %struct.B, %struct.B* %x, i32 0, i32 0, i32 0
// CHECK-NEXT:   [[VBPTR:%.*]] = getelementptr inbounds i8, i8* [[VOIDP]], i32 4
// CHECK-NEXT:   [[VBPTRPTR:%.*]] = bitcast i8* [[VBPTR:%.*]] to i32**
// CHECK-NEXT:   [[VBTBL:%.*]] = load i32*, i32** [[VBPTRPTR]], align 4
// CHECK-NEXT:   [[VBOFFP:%.*]] = getelementptr inbounds i32, i32* [[VBTBL]], i32 1
// CHECK-NEXT:   [[VBOFFS:%.*]] = load i32, i32* [[VBOFFP]], align 4
// CHECK-NEXT:   [[DELTA:%.*]] = add nsw i32 [[VBOFFS]], 4
// CHECK-NEXT:   [[ADJ:%.*]] = getelementptr inbounds i8, i8* [[VOIDP]], i32 [[DELTA]]
// CHECK-NEXT:   [[CALL:%.*]] = tail call i8* @__RTDynamicCast(i8* [[ADJ]], i32 [[DELTA]], i8* bitcast (%rtti.TypeDescriptor7* @"??_R0?AUB@@@8" to i8*), i8* bitcast (%rtti.TypeDescriptor7* @"??_R0?AUT@@@8" to i8*), i32 1)
// CHECK-NEXT:   [[RET:%.*]] = bitcast i8* [[CALL]] to %struct.T*
// CHECK-NEXT:   ret %struct.T* [[RET]]

T* test4(V* x) { return dynamic_cast<T*>(x); }
// CHECK-LABEL: define dso_local noundef %struct.T* @"?test4@@YAPAUT@@PAUV@@@Z"(%struct.V* noundef %x)
// CHECK:        [[CAST:%.*]] = bitcast %struct.V* %x to i8*
// CHECK-NEXT:   [[CALL:%.*]] = tail call i8* @__RTDynamicCast(i8* [[CAST]], i32 0, i8* bitcast (%rtti.TypeDescriptor7* @"??_R0?AUV@@@8" to i8*), i8* bitcast (%rtti.TypeDescriptor7* @"??_R0?AUT@@@8" to i8*), i32 0)
// CHECK-NEXT:   [[RET:%.*]] = bitcast i8* [[CALL]] to %struct.T*
// CHECK-NEXT:   ret %struct.T* [[RET]]

T* test5(A* x) { return dynamic_cast<T*>(x); }
// CHECK-LABEL: define dso_local noundef %struct.T* @"?test5@@YAPAUT@@PAUA@@@Z"(%struct.A* noundef %x)
// CHECK:        [[CHECK:%.*]] = icmp eq %struct.A* %x, null
// CHECK-NEXT:   br i1 [[CHECK]]
// CHECK:        [[VOIDP:%.*]] = bitcast %struct.A* %x to i8*
// CHECK-NEXT:   [[VBPTRPTR:%.*]] = getelementptr %struct.A, %struct.A* %x, i32 0, i32 0
// CHECK-NEXT:   [[VBTBL:%.*]] = load i32*, i32** [[VBPTRPTR]], align 4
// CHECK-NEXT:   [[VBOFFP:%.*]] = getelementptr inbounds i32, i32* [[VBTBL]], i32 1
// CHECK-NEXT:   [[VBOFFS:%.*]] = load i32, i32* [[VBOFFP]], align 4
// CHECK-NEXT:   [[ADJ:%.*]] = getelementptr inbounds i8, i8* [[VOIDP]], i32 [[VBOFFS]]
// CHECK-NEXT:   [[CALL:%.*]] = tail call i8* @__RTDynamicCast(i8* nonnull [[ADJ]], i32 [[VBOFFS]], i8* {{.*}}bitcast (%rtti.TypeDescriptor7* @"??_R0?AUA@@@8" to i8*), i8* {{.*}}bitcast (%rtti.TypeDescriptor7* @"??_R0?AUT@@@8" to i8*), i32 0)
// CHECK-NEXT:   [[RES:%.*]] = bitcast i8* [[CALL]] to %struct.T*
// CHECK-NEXT:   br label
// CHECK:        [[RET:%.*]] = phi %struct.T*
// CHECK-NEXT:   ret %struct.T* [[RET]]

T* test6(B* x) { return dynamic_cast<T*>(x); }
// CHECK-LABEL: define dso_local noundef %struct.T* @"?test6@@YAPAUT@@PAUB@@@Z"(%struct.B* noundef %x)
// CHECK:        [[CHECK:%.*]] = icmp eq %struct.B* %x, null
// CHECK-NEXT:   br i1 [[CHECK]]
// CHECK:        [[CAST:%.*]] = getelementptr %struct.B, %struct.B* %x, i32 0, i32 0, i32 0
// CHECK-NEXT:   [[VBPTR:%.*]] = getelementptr inbounds i8, i8* [[CAST]], i32 4
// CHECK-NEXT:   [[VBPTRPTR:%.*]] = bitcast i8* [[VBPTR]] to i32**
// CHECK-NEXT:   [[VBTBL:%.*]] = load i32*, i32** [[VBPTRPTR]], align 4
// CHECK-NEXT:   [[VBOFFP:%.*]] = getelementptr inbounds i32, i32* [[VBTBL]], i32 1
// CHECK-NEXT:   [[VBOFFS:%.*]] = load i32, i32* [[VBOFFP]], align 4
// CHECK-NEXT:   [[DELTA:%.*]] = add nsw i32 [[VBOFFS]], 4
// CHECK-NEXT:   [[ADJ:%.*]] = getelementptr inbounds i8, i8* [[CAST]], i32 [[DELTA]]
// CHECK-NEXT:   [[CALL:%.*]] = tail call i8* @__RTDynamicCast(i8* [[ADJ]], i32 [[DELTA]], i8* {{.*}}bitcast (%rtti.TypeDescriptor7* @"??_R0?AUB@@@8" to i8*), i8* {{.*}}bitcast (%rtti.TypeDescriptor7* @"??_R0?AUT@@@8" to i8*), i32 0)
// CHECK-NEXT:   [[RES:%.*]] = bitcast i8* [[CALL]] to %struct.T*
// CHECK-NEXT:   br label
// CHECK:        [[RET:%.*]] = phi %struct.T*
// CHECK-NEXT:   ret %struct.T* [[RET]]

void* test7(V* x) { return dynamic_cast<void*>(x); }
// CHECK-LABEL: define dso_local noundef i8* @"?test7@@YAPAXPAUV@@@Z"(%struct.V* noundef %x)
// CHECK:        [[CAST:%.*]] = bitcast %struct.V* %x to i8*
// CHECK-NEXT:   [[RET:%.*]] = tail call i8* @__RTCastToVoid(i8* [[CAST]])
// CHECK-NEXT:   ret i8* [[RET]]

void* test8(A* x) { return dynamic_cast<void*>(x); }
// CHECK-LABEL: define dso_local noundef i8* @"?test8@@YAPAXPAUA@@@Z"(%struct.A* noundef %x)
// CHECK:        [[CHECK:%.*]] = icmp eq %struct.A* %x, null
// CHECK-NEXT:   br i1 [[CHECK]]
// CHECK:        [[VOIDP:%.*]] = bitcast %struct.A* %x to i8*
// CHECK-NEXT:   [[VBPTRPTR:%.*]] = getelementptr %struct.A, %struct.A* %x, i32 0, i32 0
// CHECK-NEXT:   [[VBTBL:%.*]] = load i32*, i32** [[VBPTRPTR]], align 4
// CHECK-NEXT:   [[VBOFFP:%.*]] = getelementptr inbounds i32, i32* [[VBTBL]], i32 1
// CHECK-NEXT:   [[VBOFFS:%.*]] = load i32, i32* [[VBOFFP]], align 4
// CHECK-NEXT:   [[ADJ:%.*]] = getelementptr inbounds i8, i8* [[VOIDP]], i32 [[VBOFFS]]
// CHECK-NEXT:   [[RES:%.*]] = tail call i8* @__RTCastToVoid(i8* nonnull [[ADJ]])
// CHECK-NEXT:   br label
// CHECK:        [[RET:%.*]] = phi i8*
// CHECK-NEXT:   ret i8* [[RET]]

void* test9(B* x) { return dynamic_cast<void*>(x); }
// CHECK-LABEL: define dso_local noundef i8* @"?test9@@YAPAXPAUB@@@Z"(%struct.B* noundef %x)
// CHECK:        [[CHECK:%.*]] = icmp eq %struct.B* %x, null
// CHECK-NEXT:   br i1 [[CHECK]]
// CHECK:        [[CAST:%.*]] = getelementptr %struct.B, %struct.B* %x, i32 0, i32 0, i32 0
// CHECK-NEXT:   [[VBPTR:%.*]] = getelementptr inbounds i8, i8* [[CAST]], i32 4
// CHECK-NEXT:   [[VBPTRPTR:%.*]] = bitcast i8* [[VBPTR]] to i32**
// CHECK-NEXT:   [[VBTBL:%.*]] = load i32*, i32** [[VBPTRPTR]], align 4
// CHECK-NEXT:   [[VBOFFP:%.*]] = getelementptr inbounds i32, i32* [[VBTBL]], i32 1
// CHECK-NEXT:   [[VBOFFS:%.*]] = load i32, i32* [[VBOFFP]], align 4
// CHECK-NEXT:   [[DELTA:%.*]] = add nsw i32 [[VBOFFS]], 4
// CHECK-NEXT:   [[ADJ:%.*]] = getelementptr inbounds i8, i8* [[CAST]], i32 [[DELTA]]
// CHECK-NEXT:   [[CALL:%.*]] = tail call i8* @__RTCastToVoid(i8* [[ADJ]])
// CHECK-NEXT:   br label
// CHECK:        [[RET:%.*]] = phi i8*
// CHECK-NEXT:   ret i8* [[RET]]

namespace PR25606 {
struct Cleanup {
  ~Cleanup();
};
struct S1 { virtual ~S1(); };
struct S2 : virtual S1 {};
struct S3 : S2 {};

S3 *f(S2 &s) {
  Cleanup c;
  return dynamic_cast<S3 *>(&s);
}
// CHECK-LABEL: define dso_local noundef %"struct.PR25606::S3"* @"?f@PR25606@@YAPAUS3@1@AAUS2@1@@Z"(
// CHECK:    [[CALL:%.*]] = invoke i8* @__RTDynamicCast

// CHECK:    [[BC:%.*]] = bitcast i8* [[CALL]] to %"struct.PR25606::S3"*
// CHECK:    call x86_thiscallcc void @"??1Cleanup@PR25606@@QAE@XZ"(
// CHECK:    ret %"struct.PR25606::S3"* [[BC]]
}
