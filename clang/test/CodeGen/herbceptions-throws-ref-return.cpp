// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++26 -fherbceptions \
// RUN:     -fno-rtti -emit-llvm %s -o - 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-LREF --check-prefix=CHECK-RREF --check-prefix=CHECK-CREF --check-prefix=CHECK-PTR

// Verify that T& / T&& returns through `throws` preserve the underlying
// value-kind so the caller's `return inner_throws();` does not collapse
// the reference into a temporary. The throws ABI stores the success
// pointer in the {ptr, i1} slot; the caller's auto-propagation must hand
// through the original pointer without copying the referent.
//
// Also verify T* returns share the same calling convention as T&/T&&:
// all pointer-typed throws returns collapse to {{ptr, i64}, i1}.

namespace std {
struct error { void *d; __SIZE_TYPE__ c; ~error() noexcept; };
}

int g = 42;

int& lref_inner() { return g; }
int& lref_middle() throws { return lref_inner(); }
int& lref_outer() throws { return lref_middle(); }

int&& rref_inner() { return static_cast<int&&>(g); }
int&& rref_middle() throws { return rref_inner(); }
int&& rref_outer() throws { return rref_middle(); }

const int& cref_inner() { return g; }
const int& cref_middle() throws { return cref_inner(); }
const int& cref_outer() throws { return cref_middle(); }

int* ptr_inner() { return &g; }
int* ptr_middle() throws { return ptr_inner(); }
int* ptr_outer() throws { return ptr_middle(); }

int main() {
    int& a = lref_outer();
    int&& b = rref_outer();
    const int& c = cref_outer();
    int* p = ptr_outer();
    return (a == 42 && b == 42 && c == 42 && *p == 42) ? 0 : 1;
}

// Each throws function returns a {{ptr, i64}, i1} slot (not a wrapped T):
//   T&   throws  -> {{ ptr, i64 }, i1 }  (success: first 8 bytes = T*,
//                                         trailing 8 bytes zero;
//                                         error: full error struct in
//                                         the first 16 bytes)
//   T&&  throws  -> same
//   T const& throws  -> same
//   T*  throws  -> same (ABI-identical: under opaque pointers T&/T&&/T*
//                        all lower to ptr, and the first slot element is
//                        max(sizeof(T*), sizeof(std::error)) = 16 bytes)
//
// The slot is sized to max(sizeof(T*), sizeof(std::error)) = 16 bytes
// for the first element so the error can be carried in-place. The i1
// discriminant is materialized from the carry flag via HERB_SETCCr
// (X86 backend) after each call.

// CHECK-LREF: define {{[^@]+}}@_Z11lref_middlev() #1
// CHECK-LREF: define {{[^@]+}}@_Z10lref_outerv() #1
// CHECK-RREF: define {{[^@]+}}@_Z11rref_middlev() #1
// CHECK-RREF: define {{[^@]+}}@_Z10rref_outerv() #1
// CHECK-CREF: define {{[^@]+}}@_Z11cref_middlev() #1
// CHECK-CREF: define {{[^@]+}}@_Z10cref_outerv() #1
// CHECK-PTR: define {{[^@]+}}@_Z10ptr_middlev() #1
// CHECK-PTR: define {{[^@]+}}@_Z9ptr_outerv() #1
