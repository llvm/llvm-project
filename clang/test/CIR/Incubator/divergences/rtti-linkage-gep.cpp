// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG
//
// XFAIL: *
//
// CIR generates incorrect linkage and GEP types for RTTI type info structures.
//
// RTTI type info structures should be:
// 1. Marked as linkonce_odr with comdat for proper ODR compliance
// 2. Use consistent GEP indexing type (ptr-based not byte-based)
//
// Current divergences:
// 1. CIR generates: @_ZTI7Derived = constant (missing linkonce_odr, comdat)
//    CodeGen: @_ZTI7Derived = linkonce_odr constant ... comdat
//
// 2. CIR generates: getelementptr inbounds nuw (i8, ptr @..., i64 16)
//    CodeGen: getelementptr inbounds (ptr, ptr @..., i64 2)
//    These are semantically equivalent (2 ptrs = 16 bytes) but type differs
//
// This can cause linker errors and ODR violations in multi-TU programs.

struct Base {
    virtual ~Base() {}
    virtual int get() { return 1; }
};

struct Derived : Base {
    int get() override { return 2; }
    int extra() { return 3; }
};

int test_dynamic_cast(Base* b) {
    if (Derived* d = dynamic_cast<Derived*>(b)) {
        return d->extra();
    }
    return 0;
}

// LLVM: Type info should have proper linkage
// LLVM: @_ZTI7Derived = {{.*}}constant

// OGCG: Type info should be linkonce_odr with comdat
// OGCG: @_ZTI7Derived = linkonce_odr constant {{.*}} comdat
// OGCG: getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2)
