// RUN: %clang_cc1 -emit-llvm %s -triple=x86_64-pc-windows-msvc -o - | FileCheck --check-prefixes=X64,CHECK %s
// RUN: %clang_cc1 -emit-llvm %s -triple=i386-pc-windows-msvc -o - | FileCheck --check-prefixes=X86,CHECK %s

struct Bird {
  virtual ~Bird();
};

struct Parrot : public Bird {
// X64: @[[ParrotVtable:[0-9]+]] = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4Parrot@@6B@", ptr @"??_EParrot@@UEAAPEAXI@Z"] }, comdat($"??_7Parrot@@6B@")
// X86: @[[ParrotVtable:[0-9]+]] = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4Parrot@@6B@", ptr @"??_EParrot@@UAEPAXI@Z"] }, comdat($"??_7Parrot@@6B@")
// X64: @[[Bird:[0-9]+]] = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4Bird@@6B@", ptr @"??_EBird@@UEAAPEAXI@Z"] }, comdat($"??_7Bird@@6B@")
// X86: @[[Bird:[0-9]+]] = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4Bird@@6B@", ptr @"??_EBird@@UAEPAXI@Z"] }, comdat($"??_7Bird@@6B@")
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

int i = 0;
struct HasOperatorDelete : public Bird{
~HasOperatorDelete() { }
void operator delete(void *p) { i-=2; }
void operator delete[](void *p) { i--; }
};

struct AllocatedAsArray : public Bird {

};

// Vector deleting dtor for Bird is an alias because no new Bird[] expressions
// in the TU.
// X64: @"??_EBird@@UEAAPEAXI@Z" = weak dso_local unnamed_addr alias ptr (ptr, i32), ptr @"??_GBird@@UEAAPEAXI@Z"
// X86: @"??_EBird@@UAEPAXI@Z" = weak dso_local unnamed_addr alias ptr (ptr, i32), ptr @"??_GBird@@UAEPAXI@Z"
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


template<class C>
struct S {
  void foo() { void *p = new C(); delete (C *)p; }
};

S<AllocatedAsArray[1][3]> sp;

void bar() {
  dealloc(alloc());

  JustAWeirdBird B;
  B.doSmth(38);

  Bird *p = new HasOperatorDelete[2];
  dealloc(p);

  sp.foo();
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
// X64-NEXT:   %[[COOKIEGEP:.*]] = getelementptr inbounds i8, ptr %[[LPTR]], i64 -8
// X86-NEXT:   %[[COOKIEGEP:.*]] = getelementptr inbounds i8, ptr %[[LPTR]], i32 -4
// X64-NEXT:   %[[HOWMANY:.*]] = load i64, ptr %[[COOKIEGEP]]
// X86-NEXT:   %[[HOWMANY:.*]] = load i32, ptr %[[COOKIEGEP]]
// X64-NEXT:   %[[ISNOELEM:.*]] = icmp eq i64 %2, 0
// X86-NEXT:   %[[ISNOELEM:.*]] = icmp eq i32 %2, 0
// CHECK-NEXT:   br i1 %[[ISNOELEM]], label %vdtor.nocall, label %vdtor.call
// CHECK: vdtor.nocall:
// X64-NEXT:   %[[HOWMANYBYTES:.*]] = mul i64 8, %[[HOWMANY]]
// X86-NEXT:   %[[HOWMANYBYTES:.*]] = mul i32 4, %[[HOWMANY]]
// X64-NEXT:   %[[ADDCOOKIESIZE:.*]] = add i64 %[[HOWMANYBYTES]], 8
// X86-NEXT:   %[[ADDCOOKIESIZE:.*]] = add i32 %[[HOWMANYBYTES]], 4
// X64-NEXT:   call void @"??_V@YAXPEAX_K@Z"(ptr noundef %[[COOKIEGEP]], i64 noundef %[[ADDCOOKIESIZE]])
// X86-NEXT:   call void @"??_V@YAXPAXI@Z"(ptr noundef %[[COOKIEGEP]], i32 noundef %[[ADDCOOKIESIZE]])
// CHECK-NEXT:   br label %delete.end
// CHECK: vdtor.call:
// CHECK-NEXT:   %[[VTABLE:.*]] = load ptr, ptr %[[LPTR]]
// CHECK-NEXT:   %[[FPGEP:.*]] = getelementptr inbounds ptr, ptr %[[VTABLE]], i64 0
// CHECK-NEXT:   %[[FPLOAD:.*]]  = load ptr, ptr %[[FPGEP]]
// X64-NEXT:   %[[CALL:.*]] = call noundef ptr %[[FPLOAD]](ptr noundef nonnull align 8 dereferenceable(8) %[[LPTR]], i32 noundef 3)
// X86-NEXT:   %[[CALL:.*]] = call x86_thiscallcc noundef ptr %[[FPLOAD]](ptr noundef nonnull align 4 dereferenceable(4) %[[LPTR]], i32 noundef 3)
// CHECK-NEXT:   br label %delete.end
// CHECK: delete.end:
// CHECK-NEXT:   ret void

// Definition of S::foo, check that it has vector deleting destructor call
// X64-LABEL: define linkonce_odr dso_local void @"?foo@?$S@$$BY102UAllocatedAsArray@@@@QEAAXXZ"
// X86-LABEL: define linkonce_odr dso_local x86_thiscallcc void @"?foo@?$S@$$BY102UAllocatedAsArray@@@@QAEXXZ"
// CHECK: delete.notnull:                                   ; preds = %arrayctor.cont
// CHECK-NEXT:   %[[DEL_PTR:.*]] = getelementptr inbounds [1 x [3 x %struct.AllocatedAsArray]], ptr %[[THE_ARRAY:.*]], i32 0, i32 0
// X64-NEXT:   %[[COOKIEGEP:.*]] = getelementptr inbounds i8, ptr %[[DEL_PTR]], i64 -8
// X86-NEXT:   %[[COOKIEGEP:.*]] = getelementptr inbounds i8, ptr %[[DEL_PTR]], i32 -4
// X64-NEXT:   %[[HOWMANY:.*]] = load i64, ptr %[[COOKIEGEP]]
// X86-NEXT:   %[[HOWMANY:.*]] = load i32, ptr %[[COOKIEGEP]]
// X64-NEXT:   %[[ISNOELEM:.*]] = icmp eq i64 %[[HOWMANY]], 0
// X86-NEXT:   %[[ISNOELEM:.*]] = icmp eq i32 %[[HOWMANY]], 0
// CHECK-NEXT:   br i1 %[[ISNOELEM]], label %vdtor.nocall, label %vdtor.call
// CHECK: vdtor.nocall:                                     ; preds = %delete.notnull
// X64-NEXT:   %[[HOWMANYBYTES:.*]] = mul i64 8, %[[HOWMANY]]
// X86-NEXT:   %[[HOWMANYBYTES:.*]] = mul i32 4, %[[HOWMANY]]
// X64-NEXT:   %[[ADDCOOKIESIZE:.*]] = add i64 %[[HOWMANYBYTES]], 8
// X86-NEXT:   %[[ADDCOOKIESIZE:.*]] = add i32 %[[HOWMANYBYTES]], 4
// X64-NEXT:   call void @"??_V@YAXPEAX_K@Z"(ptr noundef %[[COOKIEGEP]], i64 noundef %[[ADDCOOKIESIZE]])
// X86-NEXT:   call void @"??_V@YAXPAXI@Z"(ptr noundef %[[COOKIEGEP]], i32 noundef %[[ADDCOOKIESIZE]])
// CHECK-NEXT:   br label %delete.end
// CHECK: vdtor.call:                                       ; preds = %delete.notnull
// CHECK-NEXT:   %[[VTABLE:.*]] = load ptr, ptr %[[DEL_PTR]]
// CHECK-NEXT:   %[[FPGEP:.*]] = getelementptr inbounds ptr, ptr %[[VTABLE]], i64 0
// CHECK-NEXT:   %[[FPLOAD:.*]]  = load ptr, ptr %[[FPGEP]]
// X64-NEXT:   %[[CALL:.*]] = call noundef ptr %[[FPLOAD]](ptr noundef nonnull align 8 dereferenceable(8) %[[DEL_PTR]], i32 noundef 3)
// X86-NEXT:   %[[CALL:.*]] = call x86_thiscallcc noundef ptr %[[FPLOAD]](ptr noundef nonnull align 4 dereferenceable(4) %[[DEL_PTR]], i32 noundef 3)
// CHECK-NEXT:   br label %delete.end
// CHECK: delete.end:
// CHECK-NEXT:   ret void

// Vector dtor definition for Parrot.
// X64-LABEL: define weak dso_local noundef ptr @"??_EParrot@@UEAAPEAXI@Z"(
// X64-SAME: ptr {{.*}} %[[THIS:.*]], i32 {{.*}} %[[IMPLICIT_PARAM:.*]]) unnamed_addr
// X86-LABEL: define weak dso_local x86_thiscallcc noundef ptr @"??_EParrot@@UAEPAXI@Z"(
// X86-SAME: ptr noundef nonnull align 4 dereferenceable(4) %[[THIS:.*]], i32 noundef %[[IMPLICIT_PARAM:.*]]) unnamed_addr
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
// X64-NEXT:   %[[COOKIEGEP:.*]] = getelementptr inbounds i8, ptr %[[LTHIS]], i64 -8
// X86-NEXT:   %[[COOKIEGEP:.*]] = getelementptr inbounds i8, ptr %[[LTHIS]], i32 -4
// X64-NEXT:   %[[HOWMANY:.*]] = load i64, ptr %[[COOKIEGEP]]
// X86-NEXT:   %[[HOWMANY:.*]] = load i32, ptr %[[COOKIEGEP]]
// X64-NEXT:   %[[END:.*]] = getelementptr inbounds %struct.Parrot, ptr %[[LTHIS]], i64 %[[HOWMANY]]
// X86-NEXT:   %[[END:.*]] = getelementptr inbounds %struct.Parrot, ptr %[[LTHIS]], i32 %[[HOWMANY]]
// CHECK-NEXT:   br label %arraydestroy.body
// CHECK: arraydestroy.body:
// CHECK-NEXT:   %[[PASTELEM:.*]] = phi ptr [ %delete.end, %dtor.vector ], [ %arraydestroy.element, %arraydestroy.body ]
// X64-NEXT:   %[[CURELEM:.*]] = getelementptr inbounds %struct.Parrot, ptr %[[PASTELEM]], i64 -1
// X86-NEXT:   %[[CURELEM:.*]] = getelementptr inbounds %struct.Parrot, ptr %[[PASTELEM]], i32 -1
// X64-NEXT:   call void @"??1Parrot@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %[[CURELEM]])
// X86-NEXT:   call x86_thiscallcc void @"??1Parrot@@UAE@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %[[CURELEM]])
// CHECK-NEXT:   %[[DONE:.*]] = icmp eq ptr %[[CURELEM]], %[[LTHIS]]
// CHECK-NEXT:   br i1 %[[DONE]], label %arraydestroy.done3, label %arraydestroy.body
// CHECK: arraydestroy.done3:
// CHECK-NEXT:   br label %dtor.vector.cont
// CHECK: dtor.vector.cont:
// CHECK-NEXT:   %[[FIRSTBIT:.*]] = and i32 %[[LIP]], 1
// CHECK-NEXT:   %[[ISFIRSTBITZERO:.*]] = icmp eq i32 %[[FIRSTBIT]], 0
// CHECK-NEXT:   br i1 %[[ISFIRSTBITZERO]], label %dtor.continue, label %dtor.call_delete_after_array_destroy
// CHECK: dtor.call_delete_after_array_destroy:
// X64-NEXT:     call void @"??_V@YAXPEAX_K@Z"(ptr noundef %[[COOKIEGEP]], i64 noundef 8)
// X86-NEXT:     call void @"??_V@YAXPAXI@Z"(ptr noundef %[[COOKIEGEP]], i32 noundef 4)
// CHECK-NEXT:   br label %dtor.continue
// CHECK: dtor.scalar:
// X64-NEXT:   call void @"??1Parrot@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %[[LTHIS]])
// X86-NEXT:   call x86_thiscallcc void @"??1Parrot@@UAE@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %[[LTHIS]])
// CHECK-NEXT:   %[[FIRSTBIT:.*]] = and i32 %[[LIP]], 1
// CHECK-NEXT:   %[[ISFIRSTBITZERO:.*]] = icmp eq i32 %[[FIRSTBIT]], 0
// CHECK-NEXT:   br i1 %[[ISFIRSTBITZERO]], label %dtor.continue, label %dtor.call_delete
// CHECK: dtor.call_delete:
// X64-NEXT:     call void @"??3@YAXPEAX_K@Z"(ptr noundef %[[LTHIS]], i64 noundef 8)
// X86-NEXT:     call void @"??3@YAXPAXI@Z"(ptr noundef %[[LTHIS]], i32 noundef 4)
// CHECK-NEXT:   br label %dtor.continue
// CHECK: dtor.continue:
// CHECK-NEXT:   %[[LOADRET:.*]] = load ptr, ptr %[[RET]]
// CHECK-NEXT:   ret ptr %[[LOADRET]]

// X64: define weak dso_local noundef ptr @"??_EJustAWeirdBird@@UEAAPEAXI@Z"(
// X64-SAME: ptr noundef nonnull align 8 dereferenceable(8) %this, i32 noundef %should_call_delete)
// X86: define weak dso_local x86_thiscallcc noundef ptr @"??_EJustAWeirdBird@@UAEPAXI@Z"(
// X86-SAME: ptr noundef nonnull align 4 dereferenceable(4) %this, i32 noundef %should_call_delete) unnamed_addr

// X64-LABEL: define weak dso_local noundef ptr @"??_EHasOperatorDelete@@UEAAPEAXI@Z"
// X86-LABEL: define weak dso_local x86_thiscallcc noundef ptr @"??_EHasOperatorDelete@@UAEPAXI@Z"
// CHECK: dtor.call_delete_after_array_destroy:
// X64-NEXT: call void @"??_VHasOperatorDelete@@SAXPEAX@Z"
// X86-NEXT: call void @"??_VHasOperatorDelete@@SAXPAX@Z"
// CHECK: dtor.call_delete:
// X64-NEXT: call void @"??3HasOperatorDelete@@SAXPEAX@Z"
// X86-NEXT: call void @"??3HasOperatorDelete@@SAXPAX@Z"

// X64: define weak dso_local noundef ptr @"??_EAllocatedAsArray@@UEAAPEAXI@Z"
// X86: define weak dso_local x86_thiscallcc noundef ptr @"??_EAllocatedAsArray@@UAEPAXI@Z"
