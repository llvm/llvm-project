//RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -emit-llvm -o - %s | FileCheck %s

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
// Clang emits these as 'internal constant'. {{.*}} matches for array sizes, which is irrelevant to this test.

// These checks are for types with internal linkage, which should have a '*' prefix in the typeinfo name.
//CHECK-DAG: @_ZTSN12_GLOBAL__N_11AE = internal constant {{.*}}c"*N12_GLOBAL__N_11AE\00"
//CHECK-DAG: @_ZTSZ2t2vE1L = internal constant {{.*}}c"*Z2t2vE1L\00"
//CHECK-DAG: @_ZTSPN12_GLOBAL__N_11AE = internal constant {{.*}}c"*PN12_GLOBAL__N_11AE\00"
//CHECK-DAG: @_ZTS1BIN12_GLOBAL__N_11AEE = internal constant {{.*}}c"*1BIN12_GLOBAL__N_11AEE\00"
//CHECK-DAG: @_ZTSMN12_GLOBAL__N_11AEi = internal constant {{.*}}c"*MN12_GLOBAL__N_11AEi\00"

// These checks are for types with external linkage, which should not have a '*' prefix in the typeinfo name.
//CHECK-DAG: @_ZTS3Ext = {{.*}}constant {{.*}}c"3Ext\00"
//CHECK-DAG: @_ZTSP3Ext = {{.*}}constant {{.*}}c"P3Ext\00"
//CHECK-DAG: @_ZTSN3NS21DE = {{.*}}constant {{.*}}c"N3NS21DE\00"
//CHECK-DAG: @_ZTSP3Fwd = {{.*}}constant {{.*}}c"P3Fwd\00"