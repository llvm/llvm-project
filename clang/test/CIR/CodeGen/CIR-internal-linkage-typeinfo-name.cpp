//RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -fclangir -emit-cir -o - %s | FileCheck %s

// Check that RTTI type-name strings use a leading '*' for types that do not have externally visible Clang/C++ linkage,
// and for it to be omitted from externally visible types.

namespace std { class type_info; }

//The following types have internal linkage, so their typeinfo names should have a leading '*'
namespace { struct A {}; }
const std::type_info &t1() { return typeid(A); }

const std::type_info &t2() { struct L {}; return typeid(L); }

const std::type_info &t3() { return typeid(A*); }

template <class T> struct B {};
const std::type_info &t4() { return typeid(B<A>); }

namespace { struct C { int x; }; }
const std::type_info &t5() { return typeid(int A::*); }

//Following should not have a '*' prefix in the typeinfo name, since they have external linkage
struct Ext {};

const std::type_info &t6() { return typeid(Ext); }

const std::type_info &t7() { return typeid(Ext*); }

namespace NS2 { struct D {}; }

const std::type_info &t8() { return typeid(NS2::D); }

struct Fwd;
const std::type_info &t9() { return typeid(Fwd*); }

// The following checks emitted RTTI type-names. The global name is the ABI-mangled type-name object, 
// while the string constant is the mangled type-name itself. For types without externally visible linkage,
// Clang emits these as 'internal'. {{.*}} matches for array sizes, which is irrelevant to this test.

// These checks are for types with internal linkage, which should have a '*' prefix in the typeinfo name.
//CHECK-DAG: cir.global{{.*}}internal{{.*}}@_ZTSN12_GLOBAL__N_11AE = #cir.const_array<"*N12_GLOBAL__N_11AE"{{.*}}>
//CHECK-DAG: cir.global{{.*}}internal{{.*}}@_ZTSZ2t2vE1L = #cir.const_array<"*Z2t2vE1L"{{.*}}>
//CHECK-DAG: cir.global{{.*}}internal{{.*}}@_ZTSPN12_GLOBAL__N_11AE = #cir.const_array<"*PN12_GLOBAL__N_11AE"{{.*}}>
//CHECK-DAG: cir.global{{.*}}internal{{.*}}@_ZTS1BIN12_GLOBAL__N_11AEE = #cir.const_array<"*1BIN12_GLOBAL__N_11AEE"{{.*}}>
//CHECK-DAG: cir.global{{.*}}internal{{.*}}@_ZTSMN12_GLOBAL__N_11AEi = #cir.const_array<"*MN12_GLOBAL__N_11AEi"{{.*}}>

// These checks are for types with external linkage, which should not have a '*' prefix in the typeinfo name.
//CHECK-DAG: cir.global{{.*}}@_ZTS3Ext = #cir.const_array<"3Ext"{{.*}}>
//CHECK-DAG: cir.global{{.*}}@_ZTSP3Ext = #cir.const_array<"P3Ext"{{.*}}>
//CHECK-DAG: cir.global{{.*}}@_ZTSN3NS21DE = #cir.const_array<"N3NS21DE"{{.*}}>
//CHECK-DAG: cir.global{{.*}}@_ZTSP3Fwd = #cir.const_array<"P3Fwd"{{.*}}>