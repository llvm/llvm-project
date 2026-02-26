// RUN: %clang_cc1 -triple aarch64-linux -fexperimental-allow-pointer-field-protection-attr -fexperimental-pointer-field-protection-abi -fexperimental-pointer-field-protection-tagged -emit-llvm -o - %s | FileCheck %s

// Non-standard layout. Pointer fields are signed and discriminated by type.
struct Pointer {
  int* ptr;
private:
  int private_data;
};

void pass_pointer_callee(Pointer p);

// CHECK: define dso_local void @_Z12pass_pointerP7Pointer(
void pass_pointer(Pointer *pp) {
  // CHECK: %0 = load ptr, ptr %pp.addr, align 8
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.tmp, ptr align 8 %0, i64 16, i1 false)
  // CHECK: %1 = getelementptr inbounds i8, ptr %agg.tmp, i64 0
  // CHECK: %2 = call ptr @llvm.protected.field.ptr.p0(ptr %1, i64 36403, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS7Pointer.ptr) ]
  // CHECK: %3 = load ptr, ptr %2, align 8
  // CHECK: %4 = ptrtoint ptr %3 to i64
  // CHECK: %5 = insertvalue [2 x i64] poison, i64 %4, 0
  // CHECK: %6 = getelementptr inbounds i8, ptr %agg.tmp, i64 8
  // CHECK: %7 = load i64, ptr %6, align 8
  // CHECK: %8 = insertvalue [2 x i64] %5, i64 %7, 1
  // CHECK: call void @_Z19pass_pointer_callee7Pointer([2 x i64] %8)

  pass_pointer_callee(*pp);
}

// CHECK: define dso_local void @_Z14passed_pointer7PointerPS_([2 x i64] %p.coerce, ptr noundef %pp)
void passed_pointer(Pointer p, Pointer *pp) {
  // CHECK: %p = alloca %struct.Pointer, align 8
  // CHECK: %pp.addr = alloca ptr, align 8
  // CHECK: %0 = extractvalue [2 x i64] %p.coerce, 0
  // CHECK: %1 = getelementptr inbounds i8, ptr %p, i64 0
  // CHECK: %2 = call ptr @llvm.protected.field.ptr.p0(ptr %1, i64 36403, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS7Pointer.ptr) ]
  // CHECK: %3 = inttoptr i64 %0 to ptr
  // CHECK: store ptr %3, ptr %2, align 8
  // CHECK: %4 = extractvalue [2 x i64] %p.coerce, 1
  // CHECK: %5 = getelementptr inbounds i8, ptr %p, i64 8
  // CHECK: store i64 %4, ptr %5, align 8
  // CHECK: store ptr %pp, ptr %pp.addr, align 8
  // CHECK: %6 = load ptr, ptr %pp.addr, align 8
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %6, ptr align 8 %p, i64 12, i1 false)

  *pp = p;
}

// CHECK: define dso_local [2 x i64] @_Z14return_pointerP7Pointer(ptr noundef %pp)
Pointer return_pointer(Pointer *pp) {
  // CHECK: %retval = alloca %struct.Pointer, align 8
  // CHECK: %pp.addr = alloca ptr, align 8
  // CHECK: store ptr %pp, ptr %pp.addr, align 8
  // CHECK: %0 = load ptr, ptr %pp.addr, align 8
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %retval, ptr align 8 %0, i64 16, i1 false)
  // CHECK: %1 = getelementptr inbounds i8, ptr %retval, i64 0
  // CHECK: %2 = call ptr @llvm.protected.field.ptr.p0(ptr %1, i64 36403, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS7Pointer.ptr) ]
  // CHECK: %3 = load ptr, ptr %2, align 8
  // CHECK: %4 = ptrtoint ptr %3 to i64
  // CHECK: %5 = insertvalue [2 x i64] poison, i64 %4, 0
  // CHECK: %6 = getelementptr inbounds i8, ptr %retval, i64 8
  // CHECK: %7 = load i64, ptr %6, align 8
  // CHECK: %8 = insertvalue [2 x i64] %5, i64 %7, 1
  // CHECK: ret [2 x i64] %8

  return *pp;
}

Pointer returned_pointer_callee();

// CHECK: define dso_local void @_Z16returned_pointerP7Pointer(ptr noundef %pp)
void returned_pointer(Pointer *pp) {
  // CHECK: %pp.addr = alloca ptr, align 8
  // CHECK: %ref.tmp = alloca %struct.Pointer, align 8
  // CHECK: store ptr %pp, ptr %pp.addr, align 8
  // CHECK: %call = call [2 x i64] @_Z23returned_pointer_calleev()
  // CHECK: %0 = extractvalue [2 x i64] %call, 0
  // CHECK: %1 = getelementptr inbounds i8, ptr %ref.tmp, i64 0
  // CHECK: %2 = call ptr @llvm.protected.field.ptr.p0(ptr %1, i64 36403, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS7Pointer.ptr) ]
  // CHECK: %3 = inttoptr i64 %0 to ptr
  // CHECK: store ptr %3, ptr %2, align 8
  // CHECK: %4 = extractvalue [2 x i64] %call, 1
  // CHECK: %5 = getelementptr inbounds i8, ptr %ref.tmp, i64 8
  // CHECK: store i64 %4, ptr %5, align 8
  // CHECK: %6 = load ptr, ptr %pp.addr, align 8
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %6, ptr align 8 %ref.tmp, i64 12, i1 false)

  *pp = returned_pointer_callee();
}

union PointerUnion {
  Pointer ptr;
};

void pass_pointer_union_callee(PointerUnion pu);

// CHECK: define dso_local void @_Z18pass_pointer_unionP12PointerUnion(
void pass_pointer_union(PointerUnion *pup) {
  // CHECK: %0 = load ptr, ptr %pup.addr, align 8
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.tmp, ptr align 8 %0, i64 16, i1 false)

  // CHECK: %coerce.dive = getelementptr inbounds nuw %union.PointerUnion, ptr %agg.tmp, i32 0, i32 0
  // CHECK: %1 = load [2 x i64], ptr %coerce.dive, align 8
  // CHECK: call void @_Z25pass_pointer_union_callee12PointerUnion([2 x i64] %1)

  pass_pointer_union_callee(*pup);
}

// Manual opt into PFP, non-trivially destructible.
// Pointer fields are signed and discriminated by address.
// Trivial ABI: passed and returned by value despite being non-trivial.
struct [[clang::trivial_abi]] [[clang::pointer_field_protection]] TrivialAbiPointer {
  int *ptr;
  ~TrivialAbiPointer();
};

// CHECK: define dso_local void @_Z24pass_trivial_abi_pointer17TrivialAbiPointerPS_(ptr %p.coerce, ptr noundef %pp)
void pass_trivial_abi_pointer(TrivialAbiPointer p, TrivialAbiPointer *pp) {
  // CHECK: %p = alloca %struct.TrivialAbiPointer, align 8
  // CHECK: %pp.addr = alloca ptr, align 8
  // CHECK: %coerce.dive = getelementptr inbounds nuw %struct.TrivialAbiPointer, ptr %p, i32 0, i32 0
  // CHECK: %0 = getelementptr inbounds i8, ptr %coerce.dive, i64 0
  // CHECK: %1 = ptrtoint ptr %coerce.dive to i64
  // CHECK: %2 = call ptr @llvm.protected.field.ptr.p0(ptr %0, i64 %1, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS17TrivialAbiPointer.ptr) ]
  // CHECK: store ptr %p.coerce, ptr %2, align 8
  // CHECK: store ptr %pp, ptr %pp.addr, align 8
  // CHECK: %3 = load ptr, ptr %pp.addr, align 8
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %3, ptr align 8 %p, i64 8, i1 false)
  // CHECK: %4 = getelementptr inbounds i8, ptr %3, i64 0
  // CHECK: %5 = ptrtoint ptr %3 to i64
  // CHECK: %6 = call ptr @llvm.protected.field.ptr.p0(ptr %4, i64 %5, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS17TrivialAbiPointer.ptr) ]
  // CHECK: %7 = getelementptr inbounds i8, ptr %p, i64 0
  // CHECK: %8 = ptrtoint ptr %p to i64
  // CHECK: %9 = call ptr @llvm.protected.field.ptr.p0(ptr %7, i64 %8, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS17TrivialAbiPointer.ptr) ]
  // CHECK: %10 = load ptr, ptr %9, align 8
  // CHECK: store ptr %10, ptr %6, align 8
  // CHECK: call void @_ZN17TrivialAbiPointerD1Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8) %p)

  *pp = p;
}

// CHECK: define dso_local i64 @_Z26return_trivial_abi_pointerP17TrivialAbiPointer(ptr noundef %pp)
TrivialAbiPointer return_trivial_abi_pointer(TrivialAbiPointer *pp) {
  // CHECK: %retval = alloca %struct.TrivialAbiPointer, align 8
  // CHECK: %pp.addr = alloca ptr, align 8
  // CHECK: store ptr %pp, ptr %pp.addr, align 8
  // CHECK: %0 = load ptr, ptr %pp.addr, align 8
  // CHECK: call void @_ZN17TrivialAbiPointerC1ERKS_(ptr noundef nonnull align 8 dereferenceable(8) %retval, ptr noundef nonnull align 8 dereferenceable(8) %0)
  // CHECK: %1 = getelementptr inbounds i8, ptr %retval, i64 0
  // CHECK: %2 = ptrtoint ptr %retval to i64
  // CHECK: %3 = call ptr @llvm.protected.field.ptr.p0(ptr %1, i64 %2, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS17TrivialAbiPointer.ptr) ]
  // CHECK: %4 = load ptr, ptr %3, align 8
  // CHECK: %5 = ptrtoint ptr %4 to i64
  // CHECK: ret i64 %5

  return *pp;
}

