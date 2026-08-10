// RUN: %clang_cc1 -emit-llvm -O1 -o - -fexceptions -triple=i386-pc-win32 %s | FileCheck %s

struct S { char a; };
struct V { virtual void f(); };
struct A : virtual V {};
struct B : S, virtual V {};
struct T {};

T* test0() { return dynamic_cast<T*>((B*)0); }
// CHECK-LABEL: define dso_local noalias noundef ptr @"?test0@@YAPAUT@@XZ"()
// CHECK:   ret ptr null

T* test1(V* x) { return &dynamic_cast<T&>(*x); }
// CHECK-LABEL: define dso_local noundef ptr @"?test1@@YAPAUT@@PAUV@@@Z"(ptr noundef %x)
// CHECK:   [[CALL:%.*]] = tail call ptr @__RTDynamicCast(ptr %x, i32 0, ptr nonnull @"??_R0?AUV@@@8", ptr nonnull @"??_R0?AUT@@@8", i32 1)
// CHECK-NEXT:   ret ptr [[CALL]]

T* test2(A* x) { return &dynamic_cast<T&>(*x); }
// CHECK-LABEL: define dso_local noundef ptr @"?test2@@YAPAUT@@PAUA@@@Z"(ptr noundef %x)
// CHECK:        [[VBTBL:%.*]] = load ptr, ptr %x, align 4
// CHECK-NEXT:   [[VBOFFP:%.*]] = getelementptr inbounds nuw i8, ptr [[VBTBL]], i32 4
// CHECK-NEXT:   [[VBOFFS:%.*]] = load i32, ptr [[VBOFFP]], align 4
// CHECK-NEXT:   [[ADJ:%.*]] = getelementptr inbounds i8, ptr %x, i32 [[VBOFFS]]
// CHECK-NEXT:   [[CALL:%.*]] = tail call ptr @__RTDynamicCast(ptr nonnull [[ADJ]], i32 [[VBOFFS]], ptr nonnull @"??_R0?AUA@@@8", ptr nonnull @"??_R0?AUT@@@8", i32 1)
// CHECK-NEXT:   ret ptr [[CALL]]

T* test3(B* x) { return &dynamic_cast<T&>(*x); }
// CHECK-LABEL: define dso_local noundef ptr @"?test3@@YAPAUT@@PAUB@@@Z"(ptr noundef %x)
// CHECK:        [[VBPTR:%.*]] = getelementptr inbounds nuw i8, ptr %x, i32 4
// CHECK-NEXT:   [[VBTBL:%.*]] = load ptr, ptr [[VBPTR:%.*]], align 4
// CHECK-NEXT:   [[VBOFFP:%.*]] = getelementptr inbounds nuw i8, ptr [[VBTBL]], i32 4
// CHECK-NEXT:   [[VBOFFS:%.*]] = load i32, ptr [[VBOFFP]], align 4
// CHECK-NEXT:   [[DELTA:%.*]] = add nsw i32 [[VBOFFS]], 4
// CHECK-NEXT:   [[ADJ:%.*]] = getelementptr inbounds i8, ptr %x, i32 [[DELTA]]
// CHECK-NEXT:   [[CALL:%.*]] = tail call ptr @__RTDynamicCast(ptr [[ADJ]], i32 [[DELTA]], ptr nonnull @"??_R0?AUB@@@8", ptr nonnull @"??_R0?AUT@@@8", i32 1)
// CHECK-NEXT:   ret ptr [[CALL]]

T* test4(V* x) { return dynamic_cast<T*>(x); }
// CHECK-LABEL: define dso_local noundef ptr @"?test4@@YAPAUT@@PAUV@@@Z"(ptr noundef %x)
// CHECK:   [[CALL:%.*]] = tail call ptr @__RTDynamicCast(ptr %x, i32 0, ptr nonnull @"??_R0?AUV@@@8", ptr nonnull @"??_R0?AUT@@@8", i32 0)
// CHECK-NEXT:   ret ptr [[CALL]]

T* test5(A* x) { return dynamic_cast<T*>(x); }
// CHECK-LABEL: define dso_local noundef ptr @"?test5@@YAPAUT@@PAUA@@@Z"(ptr noundef %x)
// CHECK:        [[CHECK:%.*]] = icmp eq ptr %x, null
// CHECK-NEXT:   br i1 [[CHECK]]
// CHECK:        [[VBTBL:%.*]] = load ptr, ptr %x, align 4
// CHECK-NEXT:   [[VBOFFP:%.*]] = getelementptr inbounds nuw i8, ptr [[VBTBL]], i32 4
// CHECK-NEXT:   [[VBOFFS:%.*]] = load i32, ptr [[VBOFFP]], align 4
// CHECK-NEXT:   [[ADJ:%.*]] = getelementptr inbounds i8, ptr %x, i32 [[VBOFFS]]
// CHECK-NEXT:   [[CALL:%.*]] = tail call ptr @__RTDynamicCast(ptr nonnull [[ADJ]], i32 [[VBOFFS]], ptr {{.*}}@"??_R0?AUA@@@8", ptr {{.*}}@"??_R0?AUT@@@8", i32 0)
// CHECK-NEXT:   br label
// CHECK:        [[RET:%.*]] = phi ptr
// CHECK-NEXT:   ret ptr [[RET]]

T* test6(B* x) { return dynamic_cast<T*>(x); }
// CHECK-LABEL: define dso_local noundef ptr @"?test6@@YAPAUT@@PAUB@@@Z"(ptr noundef %x)
// CHECK:        [[CHECK:%.*]] = icmp eq ptr %x, null
// CHECK-NEXT:   br i1 [[CHECK]]
// CHECK:        [[VBPTR:%.*]] = getelementptr inbounds nuw i8, ptr %x, i32 4
// CHECK-NEXT:   [[VBTBL:%.*]] = load ptr, ptr [[VBPTR]], align 4
// CHECK-NEXT:   [[VBOFFP:%.*]] = getelementptr inbounds nuw i8, ptr [[VBTBL]], i32 4
// CHECK-NEXT:   [[VBOFFS:%.*]] = load i32, ptr [[VBOFFP]], align 4
// CHECK-NEXT:   [[DELTA:%.*]] = add nsw i32 [[VBOFFS]], 4
// CHECK-NEXT:   [[ADJ:%.*]] = getelementptr inbounds i8, ptr %x, i32 [[DELTA]]
// CHECK-NEXT:   [[CALL:%.*]] = tail call ptr @__RTDynamicCast(ptr nonnull [[ADJ]], i32 [[DELTA]], ptr {{.*}}@"??_R0?AUB@@@8", ptr {{.*}}@"??_R0?AUT@@@8", i32 0)
// CHECK-NEXT:   br label
// CHECK:        [[RET:%.*]] = phi ptr
// CHECK-NEXT:   ret ptr [[RET]]

void* test7(V* x) { return dynamic_cast<void*>(x); }
// CHECK-LABEL: define dso_local noundef ptr @"?test7@@YAPAXPAUV@@@Z"(ptr noundef %x)
// CHECK:   [[RET:%.*]] = tail call ptr @__RTCastToVoid(ptr %x)
// CHECK-NEXT:   ret ptr [[RET]]

void* test8(A* x) { return dynamic_cast<void*>(x); }
// CHECK-LABEL: define dso_local noundef ptr @"?test8@@YAPAXPAUA@@@Z"(ptr noundef %x)
// CHECK:        [[CHECK:%.*]] = icmp eq ptr %x, null
// CHECK-NEXT:   br i1 [[CHECK]]
// CHECK:        [[VBTBL:%.*]] = load ptr, ptr %x, align 4
// CHECK-NEXT:   [[VBOFFP:%.*]] = getelementptr inbounds nuw i8, ptr [[VBTBL]], i32 4
// CHECK-NEXT:   [[VBOFFS:%.*]] = load i32, ptr [[VBOFFP]], align 4
// CHECK-NEXT:   [[ADJ:%.*]] = getelementptr inbounds i8, ptr %x, i32 [[VBOFFS]]
// CHECK-NEXT:   [[RES:%.*]] = tail call ptr @__RTCastToVoid(ptr nonnull [[ADJ]])
// CHECK-NEXT:   br label
// CHECK:        [[RET:%.*]] = phi ptr
// CHECK-NEXT:   ret ptr [[RET]]

void* test9(B* x) { return dynamic_cast<void*>(x); }
// CHECK-LABEL: define dso_local noundef ptr @"?test9@@YAPAXPAUB@@@Z"(ptr noundef %x)
// CHECK:        [[CHECK:%.*]] = icmp eq ptr %x, null
// CHECK-NEXT:   br i1 [[CHECK]]
// CHECK:        [[VBPTR:%.*]] = getelementptr inbounds nuw i8, ptr %x, i32 4
// CHECK-NEXT:   [[VBTBL:%.*]] = load ptr, ptr [[VBPTR]], align 4
// CHECK-NEXT:   [[VBOFFP:%.*]] = getelementptr inbounds nuw i8, ptr [[VBTBL]], i32 4
// CHECK-NEXT:   [[VBOFFS:%.*]] = load i32, ptr [[VBOFFP]], align 4
// CHECK-NEXT:   [[BASE:%.*]] = getelementptr i8, ptr %x, i32 [[VBOFFS]]
// CHECK-NEXT:   [[ADJ:%.*]] = getelementptr i8, ptr [[BASE]], i32 4
// CHECK-NEXT:   [[CALL:%.*]] = tail call ptr @__RTCastToVoid(ptr [[ADJ]])
// CHECK-NEXT:   br label
// CHECK:        [[RET:%.*]] = phi ptr
// CHECK-NEXT:   ret ptr [[RET]]

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
// CHECK-LABEL: define dso_local noundef ptr @"?f@PR25606@@YAPAUS3@1@AAUS2@1@@Z"(
// CHECK:    [[CALL:%.*]] = invoke ptr @__RTDynamicCast

// CHECK:    call x86_thiscallcc void @"??1Cleanup@PR25606@@QAE@XZ"(
// CHECK:    ret ptr [[CALL]]
}
