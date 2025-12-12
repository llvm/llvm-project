// RUN: %clang_cc1 -emit-llvm -fms-extensions %s -triple=x86_64-pc-windows-msvc -o - | FileCheck --check-prefixes=X64,CHECK %s
// RUN: %clang_cc1 -emit-llvm -fms-extensions %s -triple=i386-pc-windows-msvc -o - | FileCheck --check-prefixes=X86,CHECK %s
// RUN: %clang_cc1 -emit-llvm -fms-extensions %s -triple=x86_64-pc-windows-msvc -fclang-abi-compat=21 -o - | FileCheck --check-prefixes=CLANG21 %s

struct Bird {
  virtual ~Bird();
};

struct Parrot : public Bird {
// X64: @[[ParrotVtable:[0-9]+]] = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4Parrot@@6B@", ptr @"??_EParrot@@UEAAPEAXI@Z"] }, comdat($"??_7Parrot@@6B@")
// X86: @[[ParrotVtable:[0-9]+]] = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4Parrot@@6B@", ptr @"??_EParrot@@UAEPAXI@Z"] }, comdat($"??_7Parrot@@6B@")
// CLANG21: @[[ParrotVtable:[0-9]+]] = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4Parrot@@6B@", ptr @"??_GParrot@@UEAAPEAXI@Z"] }, comdat($"??_7Parrot@@6B@")
// X64: @[[Bird:[0-9]+]] = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4Bird@@6B@", ptr @"??_EBird@@UEAAPEAXI@Z"] }, comdat($"??_7Bird@@6B@")
// X86: @[[Bird:[0-9]+]] = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4Bird@@6B@", ptr @"??_EBird@@UAEPAXI@Z"] }, comdat($"??_7Bird@@6B@")
// CLANG21: @[[Bird:[0-9]+]] = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4Bird@@6B@", ptr @"??_GBird@@UEAAPEAXI@Z"] }, comdat($"??_7Bird@@6B@")
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
// CLANG21-NOT: @"??_E

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

// Normal loop over the array elements for clang21 ABI
// CLANG21-LABEL: define dso_local void @"?dealloc@@YAXPEAUBird@@@Z"
// CLANG21:   %p.addr = alloca ptr
// CLANG21-NEXT:   store ptr %p, ptr %p.addr
// CLANG21-NEXT:   %0 = load ptr, ptr %p.addr
// CLANG21-NEXT:   %isnull = icmp eq ptr %0, null
// CLANG21-NEXT:   br i1 %isnull, label %delete.end2, label %delete.notnull
// CLANG21: delete.notnull:
// CLANG21-NEXT:   %1 = getelementptr inbounds i8, ptr %0, i64 -8
// CLANG21-NEXT:   %2 = load i64, ptr %1
// CLANG21-NEXT:   %delete.end = getelementptr inbounds %struct.Bird, ptr %0, i64 %2
// CLANG21-NEXT:   %arraydestroy.isempty = icmp eq ptr %0, %delete.end
// CLANG21-NEXT:   br i1 %arraydestroy.isempty, label %arraydestroy.done1, label %arraydestroy.body
// CLANG21: arraydestroy.body:
// CLANG21-NEXT:   %arraydestroy.elementPast = phi ptr [ %delete.end, %delete.notnull ], [ %arraydestroy.element, %arraydestroy.body ]
// CLANG21-NEXT:   %arraydestroy.element = getelementptr inbounds %struct.Bird, ptr %arraydestroy.elementPast, i64 -1
// CLANG21-NEXT:   call void @"??1Bird@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %arraydestroy.element)
// CLANG21-NEXT:   %arraydestroy.done = icmp eq ptr %arraydestroy.element, %0
// CLANG21-NEXT:   br i1 %arraydestroy.done, label %arraydestroy.done1, label %arraydestroy.body
// CLANG21: arraydestroy.done1:
// CLANG21-NEXT:   %3 = mul i64 8, %2
// CLANG21-NEXT:   %4 = add i64 %3, 8
// CLANG21-NEXT:   call void @"??_V@YAXPEAX_K@Z"(ptr noundef %1, i64 noundef %4)
// CLANG21-NEXT:   br label %delete.end2

// Definition of S::foo, check that it has vector deleting destructor call
// X64-LABEL: define linkonce_odr dso_local void @"?foo@?$S@$$BY102UAllocatedAsArray@@@@QEAAXXZ"
// X86-LABEL: define linkonce_odr dso_local x86_thiscallcc void @"?foo@?$S@$$BY102UAllocatedAsArray@@@@QAEXXZ"
// X64: %[[NEWCALL:.*]] = call noalias noundef nonnull ptr @"??_U@YAPEAX_K@Z"(i64 noundef 32)
// X86: %[[NEWCALL:.*]] = call noalias noundef nonnull ptr @"??_U@YAPAXI@Z"(i32 noundef 16)
// X64: %[[ARR:.*]] = getelementptr inbounds i8, ptr %[[NEWCALL]], i64 8
// X86: %[[ARR:.*]] = getelementptr inbounds i8, ptr %[[NEWCALL]], i32 4
// CHECK: store ptr %[[ARR]], ptr %[[DP:.*]]
// CHECK: %[[DEL_PTR:.*]] = load ptr, ptr %[[DP:.*]]
// CHECK: delete.notnull:
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
// CLANG21: define linkonce_odr dso_local noundef ptr @"??_GJustAWeirdBird@@UEAAPEAXI@Z"(
// X86: define weak dso_local x86_thiscallcc noundef ptr @"??_EJustAWeirdBird@@UAEPAXI@Z"(
// X86-SAME: ptr noundef nonnull align 4 dereferenceable(4) %this, i32 noundef %should_call_delete) unnamed_addr

// X64-LABEL: define weak dso_local noundef ptr @"??_EHasOperatorDelete@@UEAAPEAXI@Z"
// X86-LABEL: define weak dso_local x86_thiscallcc noundef ptr @"??_EHasOperatorDelete@@UAEPAXI@Z"
// CLANG21: define linkonce_odr dso_local noundef ptr @"??_GHasOperatorDelete@@UEAAPEAXI@Z"
// CHECK: dtor.call_delete_after_array_destroy:
// CHECK-NEXT: %[[SHOULD_CALL_GLOB_DELETE:.*]] = and i32 %should_call_delete2, 4
// CHECK-NEXT: %[[CHK:.*]] = icmp eq i32 %[[SHOULD_CALL_GLOB_DELETE]], 0
// CHECK-NEXT: br i1 %[[CHK]], label %dtor.call_class_delete_after_array_destroy, label %dtor.call_glob_delete_after_array_destroy
// CHECK: dtor.call_class_delete_after_array_destroy:
// X64-NEXT:   call void @"??_VHasOperatorDelete@@SAXPEAX@Z"(ptr noundef %2)
// X86-NEXT: call void @"??_VHasOperatorDelete@@SAXPAX@Z"
// CHECK-NEXT:   br label %dtor.continue
// CHECK: dtor.call_glob_delete_after_array_destroy:
// X64-NEXT:   call void @"??_V@YAXPEAX_K@Z"(ptr noundef %2, i64 noundef 8)
// X86-NEXT:   call void @"??_V@YAXPAXI@Z"(ptr noundef %2, i32 noundef 4)
// CHECK-NEXT:   br label %dtor.continue



struct BaseDelete1 {
  void operator delete[](void *);
};
struct BaseDelete2 {
  void operator delete[](void *);
};
struct BaseDestructor {
  BaseDestructor() {}
  virtual ~BaseDestructor() = default;
};

struct Derived : BaseDelete1, BaseDelete2, BaseDestructor {
  Derived() {}
};

void foobartest() {
    Derived *a = new Derived[10]();
    ::delete[] a;
}

// X64-LABEL: define weak dso_local noundef ptr @"??_EDerived@@UEAAPEAXI@Z"(ptr {{.*}} %this, i32 noundef %should_call_delete)
// X86-LABEL: define weak dso_local x86_thiscallcc noundef ptr @"??_EDerived@@UAEPAXI@Z"(ptr {{.*}} %this, i32 noundef %should_call_delete)
// CHECK:  %retval = alloca ptr
// CHECK-NEXT:  %should_call_delete.addr = alloca i32, align 4
// CHECK-NEXT:  %this.addr = alloca ptr
// CHECK-NEXT:  store i32 %should_call_delete, ptr %should_call_delete.addr, align 4
// CHECK-NEXT:  store ptr %this, ptr %this.addr
// CHECK-NEXT:  %this1 = load ptr, ptr %this.addr
// CHECK-NEXT:  store ptr %this1, ptr %retval
// CHECK-NEXT:  %should_call_delete2 = load i32, ptr %should_call_delete.addr, align 4
// CHECK-NEXT:  %0 = and i32 %should_call_delete2, 2
// CHECK-NEXT:  %1 = icmp eq i32 %0, 0
// CHECK-NEXT:  br i1 %1, label %dtor.scalar, label %dtor.vector
// CHECK: dtor.vector:
// X64-NEXT:  %2 = getelementptr inbounds i8, ptr %this1, i64 -8
// X86-NEXT:  %2 = getelementptr inbounds i8, ptr %this1, i32 -4
// X64-NEXT:  %3 = load i64, ptr %2
// X86-NEXT:  %3 = load i32, ptr %2
// X64-NEXT:  %delete.end = getelementptr inbounds %struct.Derived, ptr %this1, i64 %3
// X86-NEXT:  %delete.end = getelementptr inbounds %struct.Derived, ptr %this1, i32 %3
// CHECK-NEXT:  br label %arraydestroy.body
// CHECK: arraydestroy.body:
// CHECK-NEXT:  %arraydestroy.elementPast = phi ptr [ %delete.end, %dtor.vector ], [ %arraydestroy.element, %arraydestroy.body ]
// X64-NEXT:  %arraydestroy.element = getelementptr inbounds %struct.Derived, ptr %arraydestroy.elementPast, i64 -1
// X86-NEXT:  %arraydestroy.element = getelementptr inbounds %struct.Derived, ptr %arraydestroy.elementPast, i32 -1
// X64-NEXT:  call void @"??1Derived@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %arraydestroy.element)
// X86-NEXT:  call x86_thiscallcc void @"??1Derived@@UAE@XZ"(ptr noundef nonnull align 4 dereferenceable(8) %arraydestroy.element)
// CHECK-NEXT:  %arraydestroy.done = icmp eq ptr %arraydestroy.element, %this1
// CHECK-NEXT:  br i1 %arraydestroy.done, label %arraydestroy.done3, label %arraydestroy.body
// CHECK: arraydestroy.done3:
// CHECK-NEXT:  br label %dtor.vector.cont
// CHECK: dtor.vector.cont:
// CHECK-NEXT:  %4 = and i32 %should_call_delete2, 1
// CHECK-NEXT:  %5 = icmp eq i32 %4, 0
// CHECK-NEXT:  br i1 %5, label %dtor.continue, label %dtor.call_delete_after_array_destroy
// CHECK: dtor.call_delete_after_array_destroy:
// X64-NEXT:  call void @"??_V@YAXPEAX_K@Z"(ptr noundef %2, i64 noundef 16)
// X86-NEXT:  call void @"??_V@YAXPAXI@Z"(ptr noundef %2, i32 noundef 8)
// CHECK-NEXT:  br label %dtor.continue
// CHECK: dtor.scalar:
// X64-NEXT:  call void @"??1Derived@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %this1)
// X86-NEXT:  call x86_thiscallcc void @"??1Derived@@UAE@XZ"(ptr noundef nonnull align 4 dereferenceable(8) %this1)
// CHECK-NEXT:  %6 = and i32 %should_call_delete2, 1
// CHECK-NEXT:  %7 = icmp eq i32 %6, 0
// CHECK-NEXT:  br i1 %7, label %dtor.continue, label %dtor.call_delete
// CHECK: dtor.call_delete:
// X64-NEXT:  call void @"??3@YAXPEAX_K@Z"(ptr noundef %this1, i64 noundef 16)
// X86-NEXT:  call void @"??3@YAXPAXI@Z"(ptr noundef %this1, i32 noundef 8)
// CHECK-NEXT:  br label %dtor.continue
// CHECK: dtor.continue:
// CHECK-NEXT:  %8 = load ptr, ptr %retval
// CHECK-NEXT:  ret ptr %8

// X64: define weak dso_local noundef ptr @"??_EAllocatedAsArray@@UEAAPEAXI@Z"
// X86: define weak dso_local x86_thiscallcc noundef ptr @"??_EAllocatedAsArray@@UAEPAXI@Z"
// CLANG21: define linkonce_odr dso_local noundef ptr @"??_GAllocatedAsArray@@UEAAPEAXI@Z"
