// RUN: %clang_cc1 -std=c++26 -triple aarch64-linux-gnu -fptrauth-intrinsics -fptrauth-calls -emit-llvm -o - %s | FileCheck %s

typedef __SIZE_TYPE__ size_t;

#define vtable_ptrauth(...) [[clang::ptrauth_vtable_pointer(__VA_ARGS__)]]
#define ADDR_AND_TYPE_DISC  vtable_ptrauth(process_independent, address_discrimination, type_discrimination) 
#define TYPE_DISC_ONLY  vtable_ptrauth(process_independent, no_address_discrimination, type_discrimination) 

struct TYPE_DISC_ONLY NoAddrDiscPoly trivially_relocatable_if_eligible {
    NoAddrDiscPoly(const NoAddrDiscPoly&);
    virtual ~NoAddrDiscPoly();
    int *__ptrauth(1,0,1) no_addr_disc;
    int b;
};

// A simple test to ensure that we don't do anything more than the memmove
// if there's no actual reason to do so, despite being in a configuration
// where in principle such work _could_ be required
// CHECK-LABEL: define internal void @_ZL4testI14NoAddrDiscPolyEvPvS1_m(
// CHECK: call void @llvm.memmove.p0.p0.i64(ptr align 8 %2, ptr align 8 %3, i64 %5, i1 false)
// CHECK-NEXT: ret void

struct ADDR_AND_TYPE_DISC AddrDiscPoly trivially_relocatable_if_eligible {
    AddrDiscPoly(const AddrDiscPoly&);
    virtual ~AddrDiscPoly();
    int *__ptrauth(1,0,1) no_addr_disc;
    int b;
};

// CHECK-LABEL: define internal void @_ZL4testI12AddrDiscPolyEvPvS1_m(
// CHECK: [[DST_PTR:%.*]] = load ptr, ptr %dest, align 8
// CHECK: [[SRC_PTR:%.*]] = load ptr, ptr %source, align 8
// CHECK: [[COUNT:%.*]] = load i64, ptr %count.addr, align 8
// CHECK: [[SIZE:%.*]] = mul i64 [[COUNT]], 24
// CHECK: call void @llvm.memmove.p0.p0.i64(ptr{{.*}}[[DST_PTR]], ptr{{.*}}[[SRC_PTR]], i64 [[SIZE]], i1 false)
// CHECK: br label %[[COPY_BODY:[a-zA-Z._]+]]
// CHECK: [[COPY_BODY]]:
// CHECK-NEXT: [[INDEX:%.*]] = phi i64 [ 0, %entry ], [ [[NEXT_INDEX:%.*]], %[[COPY_BODY]] ]
// CHECK: [[DST_OBJ:%.*]] = getelementptr inbounds %struct.AddrDiscPoly, ptr [[DST_PTR]], i64 [[INDEX]]
// CHECK: [[SRC_OBJ:%.*]] = getelementptr inbounds %struct.AddrDiscPoly, ptr [[SRC_PTR]], i64 [[INDEX]]
// CHECK: [[FIXUP_DST_ADDR:%.*]] = getelementptr inbounds i8, ptr [[DST_OBJ]], i64 0
// CHECK: [[FIXUP_DST_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_DST_ADDR]] to i64
// CHECK: [[FIXUP_DST_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_DST_ADDR_INT]], i64 [[TYPE_DISC:49645]])
// CHECK: [[FIXUP_SRC_ADDR:%.*]] = getelementptr inbounds i8, ptr [[SRC_OBJ]], i64 0
// CHECK: [[FIXUP_SRC_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_SRC_ADDR]] to i64
// CHECK: [[FIXUP_SRC_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_SRC_ADDR_INT]], i64 [[TYPE_DISC]])
// CHECK: [[PREFIXUP_VALUE:%.*]] = load ptr, ptr [[FIXUP_DST_ADDR]]
// CHECK: [[PREFIXUP_VALUE_INT:%.*]] = ptrtoint ptr [[PREFIXUP_VALUE]] to i64
// CHECK: [[FIXEDUP_VALUE_INT:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[PREFIXUP_VALUE_INT]], i32 2, i64 [[FIXUP_SRC_DISC]], i32 2, i64 [[FIXUP_DST_DISC]])
// CHECK: [[FIXEDUP_VALUE:%.*]] = inttoptr i64 [[FIXEDUP_VALUE_INT]] to ptr
// CHECK: store ptr [[FIXEDUP_VALUE]], ptr [[FIXUP_DST_ADDR]]
// CHECK: [[NEXT_INDEX:%.*]] = add nuw i64 [[INDEX]], 1
// CHECK: [[IS_COMPLETE:%.*]] = icmp eq i64 [[NEXT_INDEX]], [[COUNT]]
// CHECK: br i1 [[IS_COMPLETE]], label %[[FIXUP_END:[A-Za-z._]*]], label %[[COPY_BODY]]
// CHECK: [[FIXUP_END]]:
// CHECK-NEXT: ret void

struct ADDR_AND_TYPE_DISC A trivially_relocatable_if_eligible {
    virtual ~A();
    int i;
};

struct ADDR_AND_TYPE_DISC B trivially_relocatable_if_eligible {
    virtual ~B();
    int j;
};

struct ADDR_AND_TYPE_DISC C trivially_relocatable_if_eligible {
    virtual ~C();
    int k;
};

struct ADDR_AND_TYPE_DISC D trivially_relocatable_if_eligible {
    virtual ~D();
    int l;
};

// Though different types, the structure of MultipleBaseClasses1
// and MultipleBaseClasses1 is actually identical
struct MultipleBaseClasses1 trivially_relocatable_if_eligible : A, B {
    C c;
    D d;
};

// CHECK-LABEL: define internal void @_ZL4testI20MultipleBaseClasses1EvPvS1_m(
// CHECK: [[DST_PTR:%.*]] = load ptr, ptr %dest, align 8
// CHECK: [[SRC_PTR:%.*]] = load ptr, ptr %source, align 8
// CHECK: [[COUNT:%.*]] = load i64, ptr %count.addr, align 8
// CHECK: [[SIZE:%.*]] = mul i64 [[COUNT]], 64
// CHECK: call void @llvm.memmove.p0.p0.i64(ptr{{.*}}[[DST_PTR]], ptr{{.*}}[[SRC_PTR]], i64 [[SIZE]], i1 false)
// CHECK: br label %[[COPY_BODY:[a-zA-Z._]+]]
// CHECK: [[COPY_BODY]]:
// CHECK-NEXT: [[INDEX:%.*]] = phi i64 [ 0, %entry ], [ [[NEXT_INDEX:%.*]], %[[COPY_BODY]] ]
// CHECK: [[DST_OBJ:%.*]] = getelementptr inbounds %struct.MultipleBaseClasses1, ptr [[DST_PTR]], i64 [[INDEX]]
// CHECK: [[SRC_OBJ:%.*]] = getelementptr inbounds %struct.MultipleBaseClasses1, ptr [[SRC_PTR]], i64 [[INDEX]]

// Fixup 1: MultipleBaseClasses1::A vtable pointer
// CHECK: [[FIXUP_DST_ADDR:%.*]] = getelementptr inbounds i8, ptr [[DST_OBJ]], i64 0
// CHECK: [[FIXUP_DST_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_DST_ADDR]] to i64
// CHECK: [[FIXUP_DST_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_DST_ADDR_INT]], i64 [[TYPE_DISC:62866]])
// CHECK: [[FIXUP_SRC_ADDR:%.*]] = getelementptr inbounds i8, ptr [[SRC_OBJ]], i64 0
// CHECK: [[FIXUP_SRC_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_SRC_ADDR]] to i64
// CHECK: [[FIXUP_SRC_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_SRC_ADDR_INT]], i64 [[TYPE_DISC]])
// CHECK: [[PREFIXUP_VALUE:%.*]] = load ptr, ptr [[FIXUP_DST_ADDR]]
// CHECK: [[PREFIXUP_VALUE_INT:%.*]] = ptrtoint ptr [[PREFIXUP_VALUE]] to i64
// CHECK: [[FIXEDUP_VALUE_INT:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[PREFIXUP_VALUE_INT]], i32 2, i64 [[FIXUP_SRC_DISC]], i32 2, i64 [[FIXUP_DST_DISC]])
// CHECK: [[FIXEDUP_VALUE:%.*]] = inttoptr i64 [[FIXEDUP_VALUE_INT]] to ptr
// CHECK: store ptr [[FIXEDUP_VALUE]], ptr [[FIXUP_DST_ADDR]]

// Fixup 2: MultipleBaseClasses1::B vtable pointer
// CHECK: [[FIXUP_DST_ADDR:%.*]] = getelementptr inbounds i8, ptr [[DST_OBJ]], i64 16
// CHECK: [[FIXUP_DST_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_DST_ADDR]] to i64
// CHECK: [[FIXUP_DST_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_DST_ADDR_INT]], i64 [[TYPE_DISC:28965]])
// CHECK: [[FIXUP_SRC_ADDR:%.*]] = getelementptr inbounds i8, ptr [[SRC_OBJ]], i64 16
// CHECK: [[FIXUP_SRC_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_SRC_ADDR]] to i64
// CHECK: [[FIXUP_SRC_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_SRC_ADDR_INT]], i64 [[TYPE_DISC]])
// CHECK: [[PREFIXUP_VALUE:%.*]] = load ptr, ptr [[FIXUP_DST_ADDR]]
// CHECK: [[PREFIXUP_VALUE_INT:%.*]] = ptrtoint ptr [[PREFIXUP_VALUE]] to i64
// CHECK: [[FIXEDUP_VALUE_INT:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[PREFIXUP_VALUE_INT]], i32 2, i64 [[FIXUP_SRC_DISC]], i32 2, i64 [[FIXUP_DST_DISC]])
// CHECK: [[FIXEDUP_VALUE:%.*]] = inttoptr i64 [[FIXEDUP_VALUE_INT]] to ptr
// CHECK: store ptr [[FIXEDUP_VALUE]], ptr [[FIXUP_DST_ADDR]]

// Fixup 3: MultipleBaseClasses1::c vtable pointer
// CHECK: [[FIXUP_DST_ADDR:%.*]] = getelementptr inbounds i8, ptr [[DST_OBJ]], i64 32
// CHECK: [[FIXUP_DST_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_DST_ADDR]] to i64
// CHECK: [[FIXUP_DST_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_DST_ADDR_INT]], i64 [[TYPE_DISC:20692]])
// CHECK: [[FIXUP_SRC_ADDR:%.*]] = getelementptr inbounds i8, ptr [[SRC_OBJ]], i64 32
// CHECK: [[FIXUP_SRC_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_SRC_ADDR]] to i64
// CHECK: [[FIXUP_SRC_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_SRC_ADDR_INT]], i64 [[TYPE_DISC]])
// CHECK: [[PREFIXUP_VALUE:%.*]] = load ptr, ptr [[FIXUP_DST_ADDR]]
// CHECK: [[PREFIXUP_VALUE_INT:%.*]] = ptrtoint ptr [[PREFIXUP_VALUE]] to i64
// CHECK: [[FIXEDUP_VALUE_INT:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[PREFIXUP_VALUE_INT]], i32 2, i64 [[FIXUP_SRC_DISC]], i32 2, i64 [[FIXUP_DST_DISC]])
// CHECK: [[FIXEDUP_VALUE:%.*]] = inttoptr i64 [[FIXEDUP_VALUE_INT]] to ptr
// CHECK: store ptr [[FIXEDUP_VALUE]], ptr [[FIXUP_DST_ADDR]]

// Fixup 4: MultipleBaseClasses1::d vtable pointer
// CHECK: [[FIXUP_DST_ADDR:%.*]] = getelementptr inbounds i8, ptr [[DST_OBJ]], i64 48
// CHECK: [[FIXUP_DST_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_DST_ADDR]] to i64
// CHECK: [[FIXUP_DST_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_DST_ADDR_INT]], i64 [[TYPE_DISC:46475]])
// CHECK: [[FIXUP_SRC_ADDR:%.*]] = getelementptr inbounds i8, ptr [[SRC_OBJ]], i64 48
// CHECK: [[FIXUP_SRC_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_SRC_ADDR]] to i64
// CHECK: [[FIXUP_SRC_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_SRC_ADDR_INT]], i64 [[TYPE_DISC]])
// CHECK: [[PREFIXUP_VALUE:%.*]] = load ptr, ptr [[FIXUP_DST_ADDR]]
// CHECK: [[PREFIXUP_VALUE_INT:%.*]] = ptrtoint ptr [[PREFIXUP_VALUE]] to i64
// CHECK: [[FIXEDUP_VALUE_INT:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[PREFIXUP_VALUE_INT]], i32 2, i64 [[FIXUP_SRC_DISC]], i32 2, i64 [[FIXUP_DST_DISC]])
// CHECK: [[FIXEDUP_VALUE:%.*]] = inttoptr i64 [[FIXEDUP_VALUE_INT]] to ptr
// CHECK: store ptr [[FIXEDUP_VALUE]], ptr [[FIXUP_DST_ADDR]]

// CHECK: [[NEXT_INDEX:%.*]] = add nuw i64 [[INDEX]], 1
// CHECK: [[IS_COMPLETE:%.*]] = icmp eq i64 [[NEXT_INDEX]], [[COUNT]]
// CHECK: br i1 [[IS_COMPLETE]], label %[[FIXUP_END:[A-Za-z._]*]], label %[[COPY_BODY]]
// CHECK: [[FIXUP_END]]:
// CHECK-NEXT: ret void

struct MultipleBaseClasses2 trivially_relocatable_if_eligible : A, B, C, D {
};

// An exact copy of the above with the only change being MultipleBaseClass1->MultipleBaseClass2

// CHECK-LABEL: define internal void @_ZL4testI20MultipleBaseClasses2EvPvS1_m(
// CHECK: [[DST_PTR:%.*]] = load ptr, ptr %dest, align 8
// CHECK: [[SRC_PTR:%.*]] = load ptr, ptr %source, align 8
// CHECK: [[COUNT:%.*]] = load i64, ptr %count.addr, align 8
// CHECK: [[SIZE:%.*]] = mul i64 [[COUNT]], 64
// CHECK: call void @llvm.memmove.p0.p0.i64(ptr{{.*}}[[DST_PTR]], ptr{{.*}}[[SRC_PTR]], i64 [[SIZE]], i1 false)
// CHECK: br label %[[COPY_BODY:[a-zA-Z._]+]]
// CHECK: [[COPY_BODY]]:
// CHECK-NEXT: [[INDEX:%.*]] = phi i64 [ 0, %entry ], [ [[NEXT_INDEX:%.*]], %[[COPY_BODY]] ]
// CHECK: [[DST_OBJ:%.*]] = getelementptr inbounds %struct.MultipleBaseClasses2, ptr [[DST_PTR]], i64 [[INDEX]]
// CHECK: [[SRC_OBJ:%.*]] = getelementptr inbounds %struct.MultipleBaseClasses2, ptr [[SRC_PTR]], i64 [[INDEX]]

// Fixup 1: MultipleBaseClasses1::A vtable pointer
// CHECK: [[FIXUP_DST_ADDR:%.*]] = getelementptr inbounds i8, ptr [[DST_OBJ]], i64 0
// CHECK: [[FIXUP_DST_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_DST_ADDR]] to i64
// CHECK: [[FIXUP_DST_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_DST_ADDR_INT]], i64 [[TYPE_DISC:62866]])
// CHECK: [[FIXUP_SRC_ADDR:%.*]] = getelementptr inbounds i8, ptr [[SRC_OBJ]], i64 0
// CHECK: [[FIXUP_SRC_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_SRC_ADDR]] to i64
// CHECK: [[FIXUP_SRC_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_SRC_ADDR_INT]], i64 [[TYPE_DISC]])
// CHECK: [[PREFIXUP_VALUE:%.*]] = load ptr, ptr [[FIXUP_DST_ADDR]]
// CHECK: [[PREFIXUP_VALUE_INT:%.*]] = ptrtoint ptr [[PREFIXUP_VALUE]] to i64
// CHECK: [[FIXEDUP_VALUE_INT:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[PREFIXUP_VALUE_INT]], i32 2, i64 [[FIXUP_SRC_DISC]], i32 2, i64 [[FIXUP_DST_DISC]])
// CHECK: [[FIXEDUP_VALUE:%.*]] = inttoptr i64 [[FIXEDUP_VALUE_INT]] to ptr
// CHECK: store ptr [[FIXEDUP_VALUE]], ptr [[FIXUP_DST_ADDR]]

// Fixup 2: MultipleBaseClasses1::B vtable pointer
// CHECK: [[FIXUP_DST_ADDR:%.*]] = getelementptr inbounds i8, ptr [[DST_OBJ]], i64 16
// CHECK: [[FIXUP_DST_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_DST_ADDR]] to i64
// CHECK: [[FIXUP_DST_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_DST_ADDR_INT]], i64 [[TYPE_DISC:28965]])
// CHECK: [[FIXUP_SRC_ADDR:%.*]] = getelementptr inbounds i8, ptr [[SRC_OBJ]], i64 16
// CHECK: [[FIXUP_SRC_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_SRC_ADDR]] to i64
// CHECK: [[FIXUP_SRC_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_SRC_ADDR_INT]], i64 [[TYPE_DISC]])
// CHECK: [[PREFIXUP_VALUE:%.*]] = load ptr, ptr [[FIXUP_DST_ADDR]]
// CHECK: [[PREFIXUP_VALUE_INT:%.*]] = ptrtoint ptr [[PREFIXUP_VALUE]] to i64
// CHECK: [[FIXEDUP_VALUE_INT:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[PREFIXUP_VALUE_INT]], i32 2, i64 [[FIXUP_SRC_DISC]], i32 2, i64 [[FIXUP_DST_DISC]])
// CHECK: [[FIXEDUP_VALUE:%.*]] = inttoptr i64 [[FIXEDUP_VALUE_INT]] to ptr
// CHECK: store ptr [[FIXEDUP_VALUE]], ptr [[FIXUP_DST_ADDR]]

// Fixup 3: MultipleBaseClasses1::C vtable pointer
// CHECK: [[FIXUP_DST_ADDR:%.*]] = getelementptr inbounds i8, ptr [[DST_OBJ]], i64 32
// CHECK: [[FIXUP_DST_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_DST_ADDR]] to i64
// CHECK: [[FIXUP_DST_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_DST_ADDR_INT]], i64 [[TYPE_DISC:20692]])
// CHECK: [[FIXUP_SRC_ADDR:%.*]] = getelementptr inbounds i8, ptr [[SRC_OBJ]], i64 32
// CHECK: [[FIXUP_SRC_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_SRC_ADDR]] to i64
// CHECK: [[FIXUP_SRC_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_SRC_ADDR_INT]], i64 [[TYPE_DISC]])
// CHECK: [[PREFIXUP_VALUE:%.*]] = load ptr, ptr [[FIXUP_DST_ADDR]]
// CHECK: [[PREFIXUP_VALUE_INT:%.*]] = ptrtoint ptr [[PREFIXUP_VALUE]] to i64
// CHECK: [[FIXEDUP_VALUE_INT:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[PREFIXUP_VALUE_INT]], i32 2, i64 [[FIXUP_SRC_DISC]], i32 2, i64 [[FIXUP_DST_DISC]])
// CHECK: [[FIXEDUP_VALUE:%.*]] = inttoptr i64 [[FIXEDUP_VALUE_INT]] to ptr
// CHECK: store ptr [[FIXEDUP_VALUE]], ptr [[FIXUP_DST_ADDR]]

// Fixup 4: MultipleBaseClasses1::D vtable pointer
// CHECK: [[FIXUP_DST_ADDR:%.*]] = getelementptr inbounds i8, ptr [[DST_OBJ]], i64 48
// CHECK: [[FIXUP_DST_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_DST_ADDR]] to i64
// CHECK: [[FIXUP_DST_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_DST_ADDR_INT]], i64 [[TYPE_DISC:46475]])
// CHECK: [[FIXUP_SRC_ADDR:%.*]] = getelementptr inbounds i8, ptr [[SRC_OBJ]], i64 48
// CHECK: [[FIXUP_SRC_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_SRC_ADDR]] to i64
// CHECK: [[FIXUP_SRC_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_SRC_ADDR_INT]], i64 [[TYPE_DISC]])
// CHECK: [[PREFIXUP_VALUE:%.*]] = load ptr, ptr [[FIXUP_DST_ADDR]]
// CHECK: [[PREFIXUP_VALUE_INT:%.*]] = ptrtoint ptr [[PREFIXUP_VALUE]] to i64
// CHECK: [[FIXEDUP_VALUE_INT:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[PREFIXUP_VALUE_INT]], i32 2, i64 [[FIXUP_SRC_DISC]], i32 2, i64 [[FIXUP_DST_DISC]])
// CHECK: [[FIXEDUP_VALUE:%.*]] = inttoptr i64 [[FIXEDUP_VALUE_INT]] to ptr
// CHECK: store ptr [[FIXEDUP_VALUE]], ptr [[FIXUP_DST_ADDR]]

// CHECK: [[NEXT_INDEX:%.*]] = add nuw i64 [[INDEX]], 1
// CHECK: [[IS_COMPLETE:%.*]] = icmp eq i64 [[NEXT_INDEX]], [[COUNT]]
// CHECK: br i1 [[IS_COMPLETE]], label %[[FIXUP_END:[A-Za-z._]*]], label %[[COPY_BODY]]
// CHECK: [[FIXUP_END]]:
// CHECK-NEXT: ret void


struct ADDR_AND_TYPE_DISC Foo trivially_relocatable_if_eligible {
   int buffer;
   virtual ~Foo();
};

struct ADDR_AND_TYPE_DISC ArrayMember {
    Foo buffer[100];
    virtual void bar();
};

// CHECK-LABEL: define internal void @_ZL4testI11ArrayMemberEvPvS1_m(
// CHECK: [[DST_PTR:%.*]] = load ptr, ptr %dest, align 8
// CHECK: [[SRC_PTR:%.*]] = load ptr, ptr %source, align 8
// CHECK: [[COUNT:%.*]] = load i64, ptr %count.addr, align 8
// CHECK: [[SIZE:%.*]] = mul i64 [[COUNT]], 1608
// CHECK: call void @llvm.memmove.p0.p0.i64(ptr{{.*}}[[DST_PTR]], ptr{{.*}}[[SRC_PTR]], i64 [[SIZE]], i1 false)
// CHECK: br label %[[COPY_BODY:[a-zA-Z._]+]]
// CHECK: [[COPY_BODY]]:
// CHECK-NEXT: [[INDEX:%.*]] = phi i64 [ 0, %entry ], [ [[NEXT_INDEX:%.*]], %[[RELOCATION_SUBOBJECT_END:[a-zA-Z._]+]] ]
// CHECK: [[DST_OBJ:%.*]] = getelementptr inbounds %struct.ArrayMember, ptr [[DST_PTR]], i64 [[INDEX]]
// CHECK: [[SRC_OBJ:%.*]] = getelementptr inbounds %struct.ArrayMember, ptr [[SRC_PTR]], i64 [[INDEX]]
// CHECK: [[FIXUP_DST_ADDR:%.*]] = getelementptr inbounds i8, ptr [[DST_OBJ]], i64 0
// CHECK: [[FIXUP_DST_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_DST_ADDR]] to i64
// CHECK: [[FIXUP_DST_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_DST_ADDR_INT]], i64 [[TYPE_DISC:9693]])
// CHECK: [[FIXUP_SRC_ADDR:%.*]] = getelementptr inbounds i8, ptr [[SRC_OBJ]], i64 0
// CHECK: [[FIXUP_SRC_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_SRC_ADDR]] to i64
// CHECK: [[FIXUP_SRC_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_SRC_ADDR_INT]], i64 [[TYPE_DISC]])
// CHECK: [[PREFIXUP_VALUE:%.*]] = load ptr, ptr [[FIXUP_DST_ADDR]]
// CHECK: [[PREFIXUP_VALUE_INT:%.*]] = ptrtoint ptr [[PREFIXUP_VALUE]] to i64
// CHECK: [[FIXEDUP_VALUE_INT:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[PREFIXUP_VALUE_INT]], i32 2, i64 [[FIXUP_SRC_DISC]], i32 2, i64 [[FIXUP_DST_DISC]])
// CHECK: [[FIXEDUP_VALUE:%.*]] = inttoptr i64 [[FIXEDUP_VALUE_INT]] to ptr
// CHECK: store ptr [[FIXEDUP_VALUE]], ptr [[FIXUP_DST_ADDR]]
// CHECK: br label %[[SUBOBJECT_COPY_BODY:[a-zA-Z._0-9]+]]
// CHECK: [[SUBOBJECT_COPY_BODY]]:
// CHECK-NEXT: [[SUBOBJECT_INDEX:%.*]] = phi i64 [ 0, %[[COPY_BODY]] ], [ [[SUBOBJECT_NEXT_INDEX:%.*]], %[[SUBOBJECT_COPY_BODY]] ]
// CHECK: [[SUBOBJECT_DST_OBJ:%.*]] = getelementptr inbounds %struct.Foo, ptr [[DST_OBJ]], i64 [[SUBOBJECT_INDEX]]
// CHECK: [[SUBOBJECT_SRC_OBJ:%.*]] = getelementptr inbounds %struct.Foo, ptr [[SRC_OBJ]], i64 [[SUBOBJECT_INDEX]]
// CHECK: [[SUBOBJECT_FIXUP_DST_ADDR:%.*]] = getelementptr inbounds i8, ptr [[SUBOBJECT_DST_OBJ]], i64 0
// CHECK: [[SUBOBJECT_FIXUP_DST_ADDR_INT:%.*]] = ptrtoint ptr [[SUBOBJECT_FIXUP_DST_ADDR]] to i64
// CHECK: [[SUBOBJECT_FIXUP_DST_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[SUBOBJECT_FIXUP_DST_ADDR_INT]], i64 [[SUBOBJECT_TYPE_DISC:31380]])
// CHECK: [[SUBOBJECT_FIXUP_SRC_ADDR:%.*]] = getelementptr inbounds i8, ptr [[SUBOBJECT_SRC_OBJ]], i64 0
// CHECK: [[SUBOBJECT_FIXUP_SRC_ADDR_INT:%.*]] = ptrtoint ptr [[SUBOBJECT_FIXUP_SRC_ADDR]] to i64
// CHECK: [[SUBOBJECT_FIXUP_SRC_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[SUBOBJECT_FIXUP_SRC_ADDR_INT]], i64 [[SUBOBJECT_TYPE_DISC]])
// CHECK: [[SUBOBJECT_PREFIXUP_VALUE:%.*]] = load ptr, ptr [[SUBOBJECT_FIXUP_DST_ADDR]]
// CHECK: [[SUBOBJECT_PREFIXUP_VALUE_INT:%.*]] = ptrtoint ptr [[SUBOBJECT_PREFIXUP_VALUE]] to i64
// CHECK: [[SUBOBJECT_FIXEDUP_VALUE_INT:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[SUBOBJECT_PREFIXUP_VALUE_INT]], i32 2, i64 [[SUBOBJECT_FIXUP_SRC_DISC]], i32 2, i64 [[SUBOBJECT_FIXUP_DST_DISC]])
// CHECK: [[SUBOBJECT_FIXEDUP_VALUE:%.*]] = inttoptr i64 [[SUBOBJECT_FIXEDUP_VALUE_INT]] to ptr
// CHECK: store ptr [[SUBOBJECT_FIXEDUP_VALUE]], ptr [[SUBOBJECT_FIXUP_DST_ADDR]]

// Copying Foo[11], verifying behaviour of array copies
// CHECK-LABEL: define internal void @_ZL4testIA11_3FooEvPvS2_m(
// CHECK: [[DST_PTR:%.*]] = load ptr, ptr %dest, align 8
// CHECK: [[SRC_PTR:%.*]] = load ptr, ptr %source, align 8
// CHECK: [[INIT_COUNT:%.*]] = load i64, ptr %count.addr, align 8
// CHECK: [[SIZE:%.*]] = mul i64 [[INIT_COUNT]], 176
// CHECK: call void @llvm.memmove.p0.p0.i64(ptr{{.*}}[[DST_PTR]], ptr{{.*}}[[SRC_PTR]], i64 [[SIZE]], i1 false)
// CHECK: [[COUNT:%.*]] = mul i64 [[INIT_COUNT]], 11
// CHECK: br label %[[COPY_BODY:[a-zA-Z._]+]]
// CHECK: [[COPY_BODY]]:
// CHECK-NEXT: [[INDEX:%.*]] = phi i64 [ 0, %entry ], [ [[NEXT_INDEX:%.*]], %[[COPY_BODY]] ]
// CHECK: [[DST_OBJ:%.*]] = getelementptr inbounds %struct.Foo, ptr [[DST_PTR]], i64 [[INDEX]]
// CHECK: [[SRC_OBJ:%.*]] = getelementptr inbounds %struct.Foo, ptr [[SRC_PTR]], i64 [[INDEX]]
// CHECK: [[FIXUP_DST_ADDR:%.*]] = getelementptr inbounds i8, ptr [[DST_OBJ]], i64 0
// CHECK: [[FIXUP_DST_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_DST_ADDR]] to i64
// CHECK: [[FIXUP_DST_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_DST_ADDR_INT]], i64 [[TYPE_DISC:31380]])
// CHECK: [[FIXUP_SRC_ADDR:%.*]] = getelementptr inbounds i8, ptr [[SRC_OBJ]], i64 0
// CHECK: [[FIXUP_SRC_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_SRC_ADDR]] to i64
// CHECK: [[FIXUP_SRC_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_SRC_ADDR_INT]], i64 [[TYPE_DISC]])
// CHECK: [[PREFIXUP_VALUE:%.*]] = load ptr, ptr [[FIXUP_DST_ADDR]]
// CHECK: [[PREFIXUP_VALUE_INT:%.*]] = ptrtoint ptr [[PREFIXUP_VALUE]] to i64
// CHECK: [[FIXEDUP_VALUE_INT:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[PREFIXUP_VALUE_INT]], i32 2, i64 [[FIXUP_SRC_DISC]], i32 2, i64 [[FIXUP_DST_DISC]])
// CHECK: [[FIXEDUP_VALUE:%.*]] = inttoptr i64 [[FIXEDUP_VALUE_INT]] to ptr
// CHECK: store ptr [[FIXEDUP_VALUE]], ptr [[FIXUP_DST_ADDR]]
// CHECK: [[NEXT_INDEX:%.*]] = add nuw i64 [[INDEX]], 1
// CHECK: [[IS_COMPLETE:%.*]] = icmp eq i64 [[NEXT_INDEX]], [[COUNT]]
// CHECK: br i1 [[IS_COMPLETE]], label %[[FIXUP_END:[A-Za-z._]*]], label %[[COPY_BODY]]
// CHECK: [[FIXUP_END]]:
// CHECK-NEXT: ret void

// Copying Foo[13][17], verifying behaviour of multidimensional array copies
// CHECK-LABEL: define internal void @_ZL4testIA13_A17_3FooEvPvS3_m(
// CHECK: [[DST_PTR:%.*]] = load ptr, ptr %dest, align 8
// CHECK: [[SRC_PTR:%.*]] = load ptr, ptr %source, align 8
// CHECK: [[INIT_COUNT:%.*]] = load i64, ptr %count.addr, align 8
// CHECK: [[SIZE:%.*]] = mul i64 [[INIT_COUNT]], 3536
// CHECK: call void @llvm.memmove.p0.p0.i64(ptr{{.*}}[[DST_PTR]], ptr{{.*}}[[SRC_PTR]], i64 [[SIZE]], i1 false)
// CHECK: [[COUNT:%.*]] = mul i64 [[INIT_COUNT]], 221
// CHECK: br label %[[COPY_BODY:[a-zA-Z._]+]]
// CHECK: [[COPY_BODY]]:
// CHECK-NEXT: [[INDEX:%.*]] = phi i64 [ 0, %entry ], [ [[NEXT_INDEX:%.*]], %[[COPY_BODY]] ]
// CHECK: [[DST_OBJ:%.*]] = getelementptr inbounds %struct.Foo, ptr [[DST_PTR]], i64 [[INDEX]]
// CHECK: [[SRC_OBJ:%.*]] = getelementptr inbounds %struct.Foo, ptr [[SRC_PTR]], i64 [[INDEX]]
// CHECK: [[FIXUP_DST_ADDR:%.*]] = getelementptr inbounds i8, ptr [[DST_OBJ]], i64 0
// CHECK: [[FIXUP_DST_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_DST_ADDR]] to i64
// CHECK: [[FIXUP_DST_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_DST_ADDR_INT]], i64 [[TYPE_DISC:31380]])
// CHECK: [[FIXUP_SRC_ADDR:%.*]] = getelementptr inbounds i8, ptr [[SRC_OBJ]], i64 0
// CHECK: [[FIXUP_SRC_ADDR_INT:%.*]] = ptrtoint ptr [[FIXUP_SRC_ADDR]] to i64
// CHECK: [[FIXUP_SRC_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FIXUP_SRC_ADDR_INT]], i64 [[TYPE_DISC]])
// CHECK: [[PREFIXUP_VALUE:%.*]] = load ptr, ptr [[FIXUP_DST_ADDR]]
// CHECK: [[PREFIXUP_VALUE_INT:%.*]] = ptrtoint ptr [[PREFIXUP_VALUE]] to i64
// CHECK: [[FIXEDUP_VALUE_INT:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[PREFIXUP_VALUE_INT]], i32 2, i64 [[FIXUP_SRC_DISC]], i32 2, i64 [[FIXUP_DST_DISC]])
// CHECK: [[FIXEDUP_VALUE:%.*]] = inttoptr i64 [[FIXEDUP_VALUE_INT]] to ptr
// CHECK: store ptr [[FIXEDUP_VALUE]], ptr [[FIXUP_DST_ADDR]]
// CHECK: [[NEXT_INDEX:%.*]] = add nuw i64 [[INDEX]], 1
// CHECK: [[IS_COMPLETE:%.*]] = icmp eq i64 [[NEXT_INDEX]], [[COUNT]]
// CHECK: br i1 [[IS_COMPLETE]], label %[[FIXUP_END:[A-Za-z._]*]], label %[[COPY_BODY]]
// CHECK: [[FIXUP_END]]:
// CHECK-NEXT: ret void

template <class T> __attribute__((noinline)) static void test(void* vDest, void* vSource, size_t count) {
    T* dest = (T*)vDest;
    T* source = (T*)vSource;
    __builtin_trivially_relocate(dest, source, count);
};

void do_tests(void *Dst, void *Src) {
    test<NoAddrDiscPoly>(Dst, Src, 10);
    test<AddrDiscPoly>(Dst, Src, 10);
    test<MultipleBaseClasses1>(Dst, Src, 10);
    test<MultipleBaseClasses2>(Dst, Src, 10);
    test<ArrayMember>(Dst, Src, 10);
    test<Foo[11]>(Dst, Src, 10);
    test<Foo[13][17]>(Dst, Src, 10);
}
