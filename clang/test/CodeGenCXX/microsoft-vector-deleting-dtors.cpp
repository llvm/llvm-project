// RUN: %clang_cc1 -emit-llvm %s -triple=x86_64-pc-windows-msvc -o - | FileCheck %s

struct Bird {
  virtual ~Bird();
};

struct Parrot : public Bird {
// CHECK: @[[ParrotVtable:[0-9]+]] = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4Parrot@@6B@", ptr @"??_EParrot@@UEAAPEAXI@Z"] }, comdat($"??_7Parrot@@6B@")
// CHECK: @[[Bird:[0-9]+]] = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4Bird@@6B@", ptr @"??_EBird@@UEAAPEAXI@Z"] }, comdat($"??_7Bird@@6B@")
  virtual ~Parrot() {}
};

Bird::~Bird() {}

// For the weird bird we first emit scalar deleting destructor, then find out
// that we need vector deleting destructor and remove the alias.
struct JustAWeirdBird {
  virtual ~JustAWeirdBird() {}

  bool doSmth(int n) {
    JustAWeirdBird *c = new JustAWeirdBird[n];

    delete[] c;
    return true;
  }
};

// Vector deleting dtor for Bird is an alias because no new Bird[] expressions
// in the TU.
// CHECK: @"??_EBird@@UEAAPEAXI@Z" = weak dso_local unnamed_addr alias ptr (ptr, i32), ptr @"??_GBird@@UEAAPEAXI@Z"
// No scalar destructor for Parrot.
// CHECK-NOT: @"??_GParrot"
// No vector destructor definition for Bird.
// CHECK-NOT: define{{.*}}@"??_EBird"
// No scalar deleting dtor for JustAWeirdBird.
// CHECK-NOT: @"??_GJustAWeirdBird"

void dealloc(Bird *p) {
  delete[] p;
}

Bird* alloc() {
  Parrot* P = new Parrot[38];
  return P;
}

void bar() {
  dealloc(alloc());

  JustAWeirdBird B;
  B.doSmth(38);
}

// CHECK-LABEL: define dso_local void @{{.*}}dealloc{{.*}}(
// CHECK-SAME: ptr noundef %[[PTR:.*]])
// CHECK: entry:
// CHECK-NEXT:   %[[PTRADDR:.*]] = alloca ptr
// CHECK-NEXT:   store ptr %[[PTR]], ptr %[[PTRADDR]]
// CHECK-NEXT:   %[[LPTR:.*]] = load ptr, ptr %[[PTRADDR]]
// CHECK-NEXT:   %[[ISNULL:.*]] = icmp eq ptr %[[LPTR]], null
// CHECK-NEXT:   br i1 %[[ISNULL]], label %delete.end, label %delete.notnull
// CHECK: delete.notnull:
// CHECK-NEXT:   %[[COOKIEGEP:.*]] = getelementptr inbounds i8, ptr %[[LPTR]], i64 -8
// CHECK-NEXT:   %[[HOWMANY:.*]] = load i64, ptr %[[COOKIEGEP]]
// CHECK-NEXT:   %[[ISNOELEM:.*]] = icmp eq i64 %2, 0
// CHECK-NEXT:   br i1 %[[ISNOELEM]], label %vdtor.nocall, label %vdtor.call
// CHECK: vdtor.nocall:
// CHECK-NEXT:   %[[HOWMANYBYTES:.*]] = mul i64 8, %[[HOWMANY]]
// CHECK-NEXT:   %[[ADDCOOKIESIZE:.*]] = add i64 %[[HOWMANYBYTES]], 8
// CHECK-NEXT:   call void @"??_V@YAXPEAX_K@Z"(ptr noundef %[[COOKIEGEP]], i64 noundef %[[ADDCOOKIESIZE]])
// CHECK-NEXT:   br label %delete.end
// CHECK: vdtor.call:
// CHECK-NEXT:   %[[VTABLE:.*]] = load ptr, ptr %[[LPTR]], align 8
// CHECK-NEXT:   %[[FPGEP:.*]] = getelementptr inbounds ptr, ptr %[[VTABLE]], i64 0
// CHECK-NEXT:   %[[FPLOAD:.*]]  = load ptr, ptr %[[FPGEP]], align 8
// CHECK-NEXT:   %[[CALL:.*]] = call noundef ptr %[[FPLOAD]](ptr noundef nonnull align 8 dereferenceable(8) %[[LPTR]], i32 noundef 3)
// CHECK-NEXT:   br label %delete.end
// CHECK: delete.end:
// CHECK-NEXT:   ret void

// Vector dtor definition for Parrot.
// CHECK-LABEL: define weak dso_local noundef ptr @"??_EParrot@@UEAAPEAXI@Z"(
// CHECK-SAME: ptr {{.*}} %[[THIS:.*]], i32 {{.*}} %[[IMPLICIT_PARAM:.*]])
// CHECK: entry:
// CHECK-NEXT:   %[[RET:.*]] = alloca ptr
// CHECK-NEXT:   %[[IPADDR:.*]] = alloca i32
// CHECK-NEXT:   %[[THISADDR:.*]] = alloca ptr
// CHECK-NEXT:   store i32 %[[IMPLICIT_PARAM]], ptr %[[IPADDR]]
// CHECK-NEXT:   store ptr %[[THIS]], ptr %[[THISADDR]]
// CHECK-NEXT:   %[[LTHIS:.*]] = load ptr, ptr %[[THISADDR]]
// CHECK-NEXT:   store ptr %[[LTHIS]], ptr %[[RET]]
// CHECK-NEXT:   %[[LIP:.*]] = load i32, ptr %[[IPADDR]]
// CHECK-NEXT:   %[[SECONDBIT:.*]] = and i32 %[[LIP]], 2
// CHECK-NEXT:   %[[ISSECONDBITZERO:.*]] = icmp eq i32 %[[SECONDBIT]], 0
// CHECK-NEXT:   br i1 %[[ISSECONDBITZERO:.*]], label %dtor.scalar, label %dtor.vector
// CHECK: dtor.vector:
// CHECK-NEXT:   %[[COOKIEGEP:.*]] = getelementptr inbounds i8, ptr %[[LTHIS]], i64 -8
// CHECK-NEXT:   %[[HOWMANY:.*]] = load i64, ptr %[[COOKIEGEP]]
// CHECK-NEXT:   %[[END:.*]] = getelementptr inbounds %struct.Parrot, ptr %[[LTHIS]], i64 %[[HOWMANY]]
// CHECK-NEXT:   br label %arraydestroy.body
// CHECK: arraydestroy.body:
// CHECK-NEXT:   %[[PASTELEM:.*]] = phi ptr [ %delete.end, %dtor.vector ], [ %arraydestroy.element, %arraydestroy.body ]
// CHECK-NEXT:   %[[CURELEM:.*]] = getelementptr inbounds %struct.Parrot, ptr %[[PASTELEM]], i64 -1
// CHECK-NEXT:   call void @"??1Parrot@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %[[CURELEM]])
// CHECK-NEXT:   %[[DONE:.*]] = icmp eq ptr %[[CURELEM]], %[[LTHIS]]
// CHECK-NEXT:   br i1 %[[DONE]], label %arraydestroy.done3, label %arraydestroy.body
// CHECK: arraydestroy.done3:
// CHECK-NEXT:   br label %dtor.vector.cont
// CHECK: dtor.vector.cont:
// CHECK-NEXT:   %[[FIRSTBIT:.*]] = and i32 %[[LIP]], 1
// CHECK-NEXT:   %[[ISFIRSTBITZERO:.*]] = icmp eq i32 %[[FIRSTBIT]], 0
// CHECK-NEXT:   br i1 %[[ISFIRSTBITZERO]], label %dtor.continue, label %dtor.call_delete_after_array_destroy
// CHECK: dtor.call_delete_after_array_destroy:
// CHECK-NEXT:   call void @"??3@YAXPEAX_K@Z"(ptr noundef %[[COOKIEGEP]], i64 noundef 8)
// CHECK-NEXT:   br label %dtor.continue
// CHECK: dtor.scalar:
// CHECK-NEXT:   call void @"??1Parrot@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %[[LTHIS]])
// CHECK-NEXT:   %[[FIRSTBIT:.*]] = and i32 %[[LIP]], 1
// CHECK-NEXT:   %[[ISFIRSTBITZERO:.*]] = icmp eq i32 %[[FIRSTBIT]], 0
// CHECK-NEXT:   br i1 %[[ISFIRSTBITZERO]], label %dtor.continue, label %dtor.call_delete
// CHECK: dtor.call_delete:
// CHECK-NEXT:   call void @"??3@YAXPEAX_K@Z"(ptr noundef %[[LTHIS]], i64 noundef 8)
// CHECK-NEXT:   br label %dtor.continue
// CHECK: dtor.continue:
// CHECK-NEXT:   %[[LOADRET:.*]] = load ptr, ptr %[[RET]], align 8
// CHECK-NEXT:   ret ptr %[[LOADRET]]

// CHECK: define weak dso_local ptr @"??_EJustAWeirdBird@@UEAAPEAXI@Z"(
// CHECK-SAME: ptr %this, i32 %should_call_delete)
