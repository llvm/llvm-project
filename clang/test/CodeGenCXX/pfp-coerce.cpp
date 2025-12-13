// RUN: %clang_cc1 -triple aarch64-linux -fexperimental-allow-pointer-field-protection-attr -fexperimental-pointer-field-protection-abi -fexperimental-pointer-field-protection-tagged -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,AARCH64 %s
// RUN: %clang_cc1 -triple x86_64-linux  -fexperimental-allow-pointer-field-protection-attr -fexperimental-pointer-field-protection-abi -fexperimental-pointer-field-protection-tagged -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,X86_64 %s

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

  // AARCH64: %2 = call ptr @llvm.protected.field.ptr.p0(ptr %1, i64 36403, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS7Pointer.ptr) ]
  // X86_64: %2 = call ptr @llvm.protected.field.ptr.p0(ptr %1, i64 51, i1 false) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS7Pointer.ptr) ]

  // CHECK: %3 = load ptr, ptr %2, align 8

  // AARCH64: %4 = ptrtoint ptr %3 to i64
  // AARCH64: %5 = insertvalue [2 x i64] poison, i64 %4, 0
  // AARCH64: %6 = getelementptr inbounds i8, ptr %agg.tmp, i64 8
  // AARCH64: %7 = load i64, ptr %6, align 8
  // AARCH64: %8 = insertvalue [2 x i64] %5, i64 %7, 1
  // AARCH64: call void @_Z19pass_pointer_callee7Pointer([2 x i64] %8)

  // X86_64: %4 = insertvalue { ptr, i32 } poison, ptr %3, 0
  // X86_64: %5 = getelementptr inbounds i8, ptr %agg.tmp, i64 8
  // X86_64: %6 = load i32, ptr %5, align 8
  // X86_64: %7 = insertvalue { ptr, i32 } %4, i32 %6, 1
  // X86_64: store { ptr, i32 } %7, ptr %agg.tmp.coerce, align 8
  // X86_64: %8 = getelementptr inbounds nuw { ptr, i32 }, ptr %agg.tmp.coerce, i32 0, i32 0
  // X86_64: %9 = load ptr, ptr %8, align 8
  // X86_64: %10 = getelementptr inbounds nuw { ptr, i32 }, ptr %agg.tmp.coerce, i32 0, i32 1
  // X86_64: %11 = load i32, ptr %10, align 8
  // X86_64: call void @_Z19pass_pointer_callee7Pointer(ptr %9, i32 %11)
  pass_pointer_callee(*pp);
}

// AARCH64: define dso_local void @_Z14passed_pointer7PointerPS_([2 x i64] %p.coerce, ptr noundef %pp)
// X86_64: define dso_local void @_Z14passed_pointer7PointerPS_(ptr %p.coerce0, i32 %p.coerce1, ptr noundef %pp)
void passed_pointer(Pointer p, Pointer *pp) {
  // AARCH64: %p = alloca %struct.Pointer, align 8
  // AARCH64: %pp.addr = alloca ptr, align 8
  // AARCH64: %0 = extractvalue [2 x i64] %p.coerce, 0
  // AARCH64: %1 = getelementptr inbounds i8, ptr %p, i64 0
  // AARCH64: %2 = call ptr @llvm.protected.field.ptr.p0(ptr %1, i64 36403, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS7Pointer.ptr) ]
  // AARCH64: %3 = inttoptr i64 %0 to ptr
  // AARCH64: store ptr %3, ptr %2, align 8
  // AARCH64: %4 = extractvalue [2 x i64] %p.coerce, 1
  // AARCH64: %5 = getelementptr inbounds i8, ptr %p, i64 8
  // AARCH64: store i64 %4, ptr %5, align 8
  // AARCH64: store ptr %pp, ptr %pp.addr, align 8
  // AARCH64: %6 = load ptr, ptr %pp.addr, align 8
  // AARCH64: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %6, ptr align 8 %p, i64 12, i1 false)

  // X86_64: %p = alloca %struct.Pointer, align 8
  // X86_64: %pp.addr = alloca ptr, align 8
  // X86_64: %0 = getelementptr inbounds nuw { ptr, i32 }, ptr %p, i32 0, i32 0
  // X86_64: store ptr %p.coerce0, ptr %0, align 8
  // X86_64: %1 = getelementptr inbounds nuw { ptr, i32 }, ptr %p, i32 0, i32 1
  // X86_64: store i32 %p.coerce1, ptr %1, align 8
  // X86_64: %2 = load %struct.Pointer, ptr %p, align 8
  // X86_64: %3 = extractvalue %struct.Pointer %2, 0
  // X86_64: %4 = getelementptr inbounds i8, ptr %p, i64 0
  // X86_64: %5 = call ptr @llvm.protected.field.ptr.p0(ptr %4, i64 51, i1 false) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS7Pointer.ptr) ]
  // X86_64: store ptr %3, ptr %5, align 8
  // X86_64: %6 = extractvalue %struct.Pointer %2, 1
  // X86_64: %7 = getelementptr inbounds i8, ptr %p, i64 8
  // X86_64: store i32 %6, ptr %7, align 8
  // X86_64: %8 = extractvalue %struct.Pointer %2, 2
  // X86_64: %9 = getelementptr inbounds i8, ptr %p, i64 12
  // X86_64: store [4 x i8] %8, ptr %9, align 4
  // X86_64: store ptr %pp, ptr %pp.addr, align 8
  // X86_64: %10 = load ptr, ptr %pp.addr, align 8
  // X86_64: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %10, ptr align 8 %p, i64 12, i1 false)
  *pp = p;
}

// AARCH64: define dso_local [2 x i64] @_Z14return_pointerP7Pointer(ptr noundef %pp)
// X86_64: define dso_local { ptr, i32 } @_Z14return_pointerP7Pointer(ptr noundef %pp)
Pointer return_pointer(Pointer *pp) {
  // AARCH64: %retval = alloca %struct.Pointer, align 8
  // AARCH64: %pp.addr = alloca ptr, align 8
  // AARCH64: store ptr %pp, ptr %pp.addr, align 8
  // AARCH64: %0 = load ptr, ptr %pp.addr, align 8
  // AARCH64: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %retval, ptr align 8 %0, i64 16, i1 false)
  // AARCH64: %1 = getelementptr inbounds i8, ptr %retval, i64 0
  // AARCH64: %2 = call ptr @llvm.protected.field.ptr.p0(ptr %1, i64 36403, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS7Pointer.ptr) ]
  // AARCH64: %3 = load ptr, ptr %2, align 8
  // AARCH64: %4 = ptrtoint ptr %3 to i64
  // AARCH64: %5 = insertvalue [2 x i64] poison, i64 %4, 0
  // AARCH64: %6 = getelementptr inbounds i8, ptr %retval, i64 8
  // AARCH64: %7 = load i64, ptr %6, align 8
  // AARCH64: %8 = insertvalue [2 x i64] %5, i64 %7, 1
  // AARCH64: ret [2 x i64] %8

  // X86_64: %retval = alloca %struct.Pointer, align 8
  // X86_64: %pp.addr = alloca ptr, align 8
  // X86_64: store ptr %pp, ptr %pp.addr, align 8
  // X86_64: %0 = load ptr, ptr %pp.addr, align 8
  // X86_64: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %retval, ptr align 8 %0, i64 16, i1 false)
  // X86_64: %1 = getelementptr inbounds i8, ptr %retval, i64 0
  // X86_64: %2 = call ptr @llvm.protected.field.ptr.p0(ptr %1, i64 51, i1 false) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS7Pointer.ptr) ]
  // X86_64: %3 = load ptr, ptr %2, align 8
  // X86_64: %4 = insertvalue { ptr, i32 } poison, ptr %3, 0
  // X86_64: %5 = getelementptr inbounds i8, ptr %retval, i64 8
  // X86_64: %6 = load i32, ptr %5, align 8
  // X86_64: %7 = insertvalue { ptr, i32 } %4, i32 %6, 1
  // X86_64: ret { ptr, i32 } %7
  return *pp;
}

Pointer returned_pointer_callee();

// CHECK: define dso_local void @_Z16returned_pointerP7Pointer(ptr noundef %pp)
void returned_pointer(Pointer *pp) {
  // AARCH64: %pp.addr = alloca ptr, align 8
  // AARCH64: %ref.tmp = alloca %struct.Pointer, align 8
  // AARCH64: store ptr %pp, ptr %pp.addr, align 8
  // AARCH64: %call = call [2 x i64] @_Z23returned_pointer_calleev()
  // AARCH64: %0 = extractvalue [2 x i64] %call, 0
  // AARCH64: %1 = getelementptr inbounds i8, ptr %ref.tmp, i64 0
  // AARCH64: %2 = call ptr @llvm.protected.field.ptr.p0(ptr %1, i64 36403, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS7Pointer.ptr) ]
  // AARCH64: %3 = inttoptr i64 %0 to ptr
  // AARCH64: store ptr %3, ptr %2, align 8
  // AARCH64: %4 = extractvalue [2 x i64] %call, 1
  // AARCH64: %5 = getelementptr inbounds i8, ptr %ref.tmp, i64 8
  // AARCH64: store i64 %4, ptr %5, align 8
  // AARCH64: %6 = load ptr, ptr %pp.addr, align 8
  // AARCH64: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %6, ptr align 8 %ref.tmp, i64 12, i1 false)

  // X86_64: %pp.addr = alloca ptr, align 8
  // X86_64: %ref.tmp = alloca %struct.Pointer, align 8
  // X86_64: store ptr %pp, ptr %pp.addr, align 8
  // X86_64: %call = call { ptr, i32 } @_Z23returned_pointer_calleev()
  // X86_64: %0 = extractvalue { ptr, i32 } %call, 0
  // X86_64: %1 = getelementptr inbounds i8, ptr %ref.tmp, i64 0
  // X86_64: %2 = call ptr @llvm.protected.field.ptr.p0(ptr %1, i64 51, i1 false) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS7Pointer.ptr) ]
  // X86_64: store ptr %0, ptr %2, align 8
  // X86_64: %3 = extractvalue { ptr, i32 } %call, 1
  // X86_64: %4 = getelementptr inbounds i8, ptr %ref.tmp, i64 8
  // X86_64: store i32 %3, ptr %4, align 8
  // X86_64: %5 = load ptr, ptr %pp.addr, align 8
  // X86_64: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %5, ptr align 8 %ref.tmp, i64 12, i1 false)
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

  // AARCH64: %coerce.dive = getelementptr inbounds nuw %union.PointerUnion, ptr %agg.tmp, i32 0, i32 0
  // AARCH64: %1 = load [2 x i64], ptr %coerce.dive, align 8
  // AARCH64: call void @_Z25pass_pointer_union_callee12PointerUnion([2 x i64] %1)

  // X86_64: %1 = getelementptr inbounds nuw { ptr, i32 }, ptr %agg.tmp, i32 0, i32 0
  // X86_64: %2 = load ptr, ptr %1, align 8
  // X86_64: %3 = getelementptr inbounds nuw { ptr, i32 }, ptr %agg.tmp, i32 0, i32 1
  // X86_64: %4 = load i32, ptr %3, align 8
  // X86_64: call void @_Z25pass_pointer_union_callee12PointerUnion(ptr %2, i32 %4)
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
  // AARCH64: %p = alloca %struct.TrivialAbiPointer, align 8
  // AARCH64: %pp.addr = alloca ptr, align 8
  // AARCH64: %coerce.dive = getelementptr inbounds nuw %struct.TrivialAbiPointer, ptr %p, i32 0, i32 0
  // AARCH64: %0 = getelementptr inbounds i8, ptr %coerce.dive, i64 0
  // AARCH64: %1 = ptrtoint ptr %coerce.dive to i64
  // AARCH64: %2 = call ptr @llvm.protected.field.ptr.p0(ptr %0, i64 %1, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS17TrivialAbiPointer.ptr) ]
  // AARCH64: store ptr %p.coerce, ptr %2, align 8
  // AARCH64: store ptr %pp, ptr %pp.addr, align 8
  // AARCH64: %3 = load ptr, ptr %pp.addr, align 8
  // AARCH64: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %3, ptr align 8 %p, i64 8, i1 false)
  // AARCH64: %4 = getelementptr inbounds i8, ptr %3, i64 0
  // AARCH64: %5 = ptrtoint ptr %3 to i64
  // AARCH64: %6 = call ptr @llvm.protected.field.ptr.p0(ptr %4, i64 %5, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS17TrivialAbiPointer.ptr) ]
  // AARCH64: %7 = getelementptr inbounds i8, ptr %p, i64 0
  // AARCH64: %8 = ptrtoint ptr %p to i64
  // AARCH64: %9 = call ptr @llvm.protected.field.ptr.p0(ptr %7, i64 %8, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS17TrivialAbiPointer.ptr) ]
  // AARCH64: %10 = load ptr, ptr %9, align 8
  // AARCH64: store ptr %10, ptr %6, align 8
  // AARCH64: call void @_ZN17TrivialAbiPointerD1Ev(ptr noundef nonnull align 8 dereferenceable(8) %p)

  // X86_64: %p = alloca %struct.TrivialAbiPointer, align 8
  // X86_64: %pp.addr = alloca ptr, align 8
  // X86_64: %coerce.dive = getelementptr inbounds nuw %struct.TrivialAbiPointer, ptr %p, i32 0, i32 0
  // X86_64: %0 = getelementptr inbounds i8, ptr %coerce.dive, i64 0
  // X86_64: %1 = call ptr @llvm.protected.field.ptr.p0(ptr %0, i64 33, i1 false) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS17TrivialAbiPointer.ptr) ]
  // X86_64: store ptr %p.coerce, ptr %1, align 8
  // X86_64: store ptr %pp, ptr %pp.addr, align 8
  // X86_64: %2 = load ptr, ptr %pp.addr, align 8
  // X86_64: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %2, ptr align 8 %p, i64 8, i1 false)
  // X86_64: call void @_ZN17TrivialAbiPointerD1Ev(ptr noundef nonnull align 8 dereferenceable(8) %p)
  *pp = p;
}

// AARCH64: define dso_local i64 @_Z26return_trivial_abi_pointerP17TrivialAbiPointer(ptr noundef %pp)
// X86_64: define dso_local ptr @_Z26return_trivial_abi_pointerP17TrivialAbiPointer(ptr noundef %pp)
TrivialAbiPointer return_trivial_abi_pointer(TrivialAbiPointer *pp) {
  // AARCH64: %retval = alloca %struct.TrivialAbiPointer, align 8
  // AARCH64: %pp.addr = alloca ptr, align 8
  // AARCH64: store ptr %pp, ptr %pp.addr, align 8
  // AARCH64: %0 = load ptr, ptr %pp.addr, align 8
  // AARCH64: call void @_ZN17TrivialAbiPointerC1ERKS_(ptr noundef nonnull align 8 dereferenceable(8) %retval, ptr noundef nonnull align 8 dereferenceable(8) %0)
  // AARCH64: %1 = getelementptr inbounds i8, ptr %retval, i64 0
  // AARCH64: %2 = ptrtoint ptr %retval to i64
  // AARCH64: %3 = call ptr @llvm.protected.field.ptr.p0(ptr %1, i64 %2, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS17TrivialAbiPointer.ptr) ]
  // AARCH64: %4 = load ptr, ptr %3, align 8
  // AARCH64: %5 = ptrtoint ptr %4 to i64
  // AARCH64: ret i64 %5

  // X86_64: %retval = alloca %struct.TrivialAbiPointer, align 8
  // X86_64: %pp.addr = alloca ptr, align 8
  // X86_64: store ptr %pp, ptr %pp.addr, align 8
  // X86_64: %0 = load ptr, ptr %pp.addr, align 8
  // X86_64: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %retval, ptr align 8 %0, i64 8, i1 false)
  // X86_64: %1 = getelementptr inbounds i8, ptr %retval, i64 0
  // X86_64: %2 = call ptr @llvm.protected.field.ptr.p0(ptr %1, i64 33, i1 false) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS17TrivialAbiPointer.ptr) ]
  // X86_64: %3 = load ptr, ptr %2, align 8
  // X86_64: ret ptr %3
  return *pp;
}

