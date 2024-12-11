
// RUN: cp %s %t
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -fblocks -verify %t
// RUN: not %clang_cc1 -fbounds-safety -fixit -fix-what-you-can %t 2> /dev/null
// RUN: grep -v FIXIT-CHECK %t | FileCheck --check-prefix=FIXIT-CHECK %s
#include <ptrcheck.h>

typedef struct F F_t;

F_t* single_source(void);

int void_ptr_single_sink(void* __single);

// The MACRO_PTR_* macros should not be modifed by any FixIt
// FIXIT-CHECK: #define MACRO_PTR_UNSAFE_TY struct F* __unsafe_indexable
#define MACRO_PTR_UNSAFE_TY struct F* __unsafe_indexable
// FIXIT-CHECK: #define MACRO_PTR_IDX_TY struct F* __indexable
#define MACRO_PTR_IDX_TY struct F* __indexable
// FIXIT-CHECK: #define MACRO_PTR_BIDI_TY struct F* __bidi_indexable
#define MACRO_PTR_BIDI_TY struct F* __bidi_indexable
// FIXIT-CHECK: #define MACRO_PTR_SINGLE_TY struct F* __single
#define MACRO_PTR_SINGLE_TY struct F* __single
// FIXIT-CHECK: #define MACRO_PTR_TY struct F*
#define MACRO_PTR_TY struct F*
#define MACRO_TY struct F

int opaque_init_assign(F_t* single_f, void* single_void,
    void* __single explicit_single_void) {
    // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oia_local' as '__single'}}
    // expected-note@+2{{pointer 'oia_local' declared here}}
    // FIXIT-CHECK: F_t* __single oia_local = single_f;
    F_t* oia_local = single_f; // Fix
    // FIXIT-CHECK: F_t* __single oia_local_nf = single_f;
    F_t* __single oia_local_nf = single_f; // No Fix

    // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oia_local2' as '__single'}}
    // expected-note@+2{{pointer 'oia_local2' declared here}}
    // FIXIT-CHECK: F_t* __single oia_local2 = single_source();
    F_t* oia_local2 = single_source(); // Fix
    // FIXIT-CHECK: F_t* __single oia_local2_nf = single_source();
    F_t* __single oia_local2_nf = single_source(); // No Fix

    // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oia_local3' as '__single'}}
    // expected-note@+2{{pointer 'oia_local3' declared here}}
    // FIXIT-CHECK: MACRO_PTR_TY __single oia_local3 = single_f;
    MACRO_PTR_TY oia_local3 = single_f; // Fix
    // FIXIT-CHECK: MACRO_PTR_TY __single oia_local3_nf = single_f;
    MACRO_PTR_TY __single oia_local3_nf = single_f; // No Fix

    // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oia_local4' as '__single'}}
    // expected-note@+2{{pointer 'oia_local4' declared here}}
    // FIXIT-CHECK: MACRO_TY* __single oia_local4 = single_f;
    MACRO_TY* oia_local4 = single_f; // Fix
    // FIXIT-CHECK: MACRO_TY* __single oia_local4_nf = single_f;
    MACRO_TY* __single oia_local4_nf = single_f; // No Fix

    // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oia_local5' as '__single'}}
    // expected-note@+2{{pointer 'oia_local5' declared here}}
    // FIXIT-CHECK: void* __single oia_local5 = single_void;
    void* oia_local5 = single_void; // Fix
    // FIXIT-CHECK: void* __single oia_local5_nf = single_void;
    void* __single oia_local5_nf = single_void; // No Fix

    // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oia_local6' as '__single'}}
    // expected-note@+2{{pointer 'oia_local6' declared here}}
    // FIXIT-CHECK: void* __single oia_local6 = explicit_single_void;
    void* oia_local6 = explicit_single_void; // Fix
    // FIXIT-CHECK: void* __single oia_local6_nf = explicit_single_void;
    void* __single oia_local6_nf = explicit_single_void; // No Fix

    // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oia_local7' as '__single'}}
    // expected-note@+2{{pointer 'oia_local7' declared here}}
    // FIXIT-CHECK: void* _Nullable __single oia_local7 = explicit_single_void;
    void* _Nullable oia_local7 = explicit_single_void; // Fix

    // No diagnostic
    // FIXIT-CHECK: MACRO_PTR_SINGLE_TY oia_local8 = single_f;
    MACRO_PTR_SINGLE_TY oia_local8 = single_f; // No Fix

    // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oia_local9' as '__single'}}
    // expected-note@+2{{pointer 'oia_local9' declared here}}}
    // FIXIT-CHECK: MACRO_PTR_BIDI_TY oia_local9 = single_f;
    MACRO_PTR_BIDI_TY oia_local9 = single_f; // No Fix

    // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oia_local10' as '__single'}}
    // expected-note@+2{{pointer 'oia_local10' declared here}}}
    // FIXIT-CHECK: MACRO_PTR_IDX_TY oia_local10 = single_f;
    MACRO_PTR_IDX_TY oia_local10 = single_f; // No Fix

    // No diagnostic
    // FIXIT-CHECK: MACRO_PTR_UNSAFE_TY oia_local11 = single_f;
    MACRO_PTR_UNSAFE_TY oia_local11 = single_f; // No Fix
}

int opaque_assign(F_t* imp_single, void* single_void,
    void* __single explicit_single_void) {
    // expected-error-re@+4{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oa_local' as '__single'}}
    // expected-note@+2{{pointer 'oa_local' declared here}}
    // FIXIT-CHECK: F_t* __single oa_local = 0;
    F_t* oa_local = 0; // Fix
    oa_local = imp_single;
    // FIXIT-CHECK: F_t* __single oa_local_nf = 0;
    F_t* __single oa_local_nf = 0; // No Fix
    oa_local_nf = imp_single;

    // expected-error-re@+4{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oa_local2' as '__single'}}
    // expected-note@+2{{pointer 'oa_local2' declared here}}
    // FIXIT-CHECK: F_t* __single oa_local2 = 0;
    F_t* oa_local2 = 0; // Fix
    oa_local2 = single_source();
    // FIXIT-CHECK: F_t* __single oa_local2_nf = 0;
    F_t* __single oa_local2_nf = 0; // No Fix
    oa_local2_nf = single_source();

    // expected-error-re@+4{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oa_local3' as '__single'}}
    // expected-note@+2{{pointer 'oa_local3' declared here}}
    // FIXIT-CHECK: MACRO_PTR_TY __single oa_local3 = 0;
    MACRO_PTR_TY oa_local3 = 0; // Fix
    oa_local3 = imp_single;
    // FIXIT-CHECK: MACRO_PTR_TY __single oa_local3_nf = 0;
    MACRO_PTR_TY __single oa_local3_nf = 0; // No Fix
    oa_local3_nf = imp_single;

    // expected-error-re@+4{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oa_local4' as '__single'}}
    // expected-note@+2{{pointer 'oa_local4' declared here}}
    // FIXIT-CHECK: MACRO_TY* __single oa_local4 = 0;
    MACRO_TY* oa_local4 = 0; // Fix
    oa_local4 = imp_single;
    // FIXIT-CHECK: MACRO_TY* __single oa_local4_nf = 0;
    MACRO_TY* __single oa_local4_nf = 0; // No Fix
    oa_local4_nf = imp_single;

    // expected-error-re@+4{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oa_local5' as '__single'}}
    // expected-note@+2{{pointer 'oa_local5' declared here}}
    // FIXIT-CHECK: void* __single oa_local5 = 0;
    void* oa_local5 = 0; // Fix
    oa_local5 = single_void;
    // FIXIT-CHECK: void* __single oa_local5_nf = 0;
    void* __single oa_local5_nf = 0; // No Fix
    oa_local5_nf = single_void;

    // expected-error-re@+4{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oa_local6' as '__single'}}
    // expected-note@+2{{pointer 'oa_local6' declared here}}
    // FIXIT-CHECK: void* __single oa_local6 = 0;
    void* oa_local6 = 0; // Fix
    oa_local6 = explicit_single_void;

    // expected-error-re@+4{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oa_local7' as '__single'}}
    // expected-note@+2{{pointer 'oa_local7' declared here}}
    // FIXIT-CHECK: void* _Nullable __single oa_local7;
    void* _Nullable oa_local7;
    oa_local7 = explicit_single_void; // Fix

    // No diagnostic
    // FIXIT-CHECK: MACRO_PTR_SINGLE_TY oa_local8;
    MACRO_PTR_SINGLE_TY oa_local8; // No Fix
    oa_local8 = imp_single;

    // expected-error-re@+4{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oa_local9' as '__single'}}
    // expected-note@+2{{pointer 'oa_local9' declared here}}}
    // FIXIT-CHECK: MACRO_PTR_BIDI_TY oa_local9;
    MACRO_PTR_BIDI_TY oa_local9; // No Fix
    oa_local9 = imp_single;

    // expected-error-re@+4{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oa_local10' as '__single'}}
    // expected-note@+2{{pointer 'oa_local10' declared here}}}
    // FIXIT-CHECK: MACRO_PTR_IDX_TY oa_local10;
    MACRO_PTR_IDX_TY oa_local10; // No Fix
    oa_local10 = imp_single;

    // No diagnostic
    // FIXIT-CHECK: MACRO_PTR_UNSAFE_TY oa_local11;
    MACRO_PTR_UNSAFE_TY oa_local11; // No Fix
    oa_local11 = imp_single;

    // Test assignment with parentheses on LHS
    // expected-error-re@+4{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oa_local12' as '__single'}}
    // expected-note@+2{{pointer 'oa_local12' declared here}}
    // FIXIT-CHECK: F_t* __single oa_local12 = 0;
    F_t* oa_local12 = 0; // Fix
    (oa_local12) = imp_single;
}

int opaque_multiple_assignments(F_t* imp_single) {
    // expected-error-re@+6{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oma_local' as '__single'}}
    // expected-note@+4{{pointer 'oma_local' declared here}}
    // expected-error-re@+5{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oma_local' as '__single'}}
    // expected-note@+2{{pointer 'oma_local' declared here}}
    // FIXIT-CHECK: F_t* __single oma_local = 0; 
    F_t* oma_local = 0; // Fix
    oma_local = imp_single;
    oma_local = single_source();

    // expected-error-re@+6{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oma_local3' as '__single'}}
    // expected-note@+4{{pointer 'oma_local3' declared here}}
    // expected-error-re@+5{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oma_local3' as '__single'}}
    // expected-note@+2{{pointer 'oma_local3' declared here}}
    // FIXIT-CHECK: MACRO_PTR_TY __single oma_local3 = 0; 
    MACRO_PTR_TY oma_local3 = 0; // Fix
    oma_local3 = imp_single;
    oma_local3 = single_source();

    // expected-error-re@+6{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oma_local4' as '__single'}}
    // expected-note@+4{{pointer 'oma_local4' declared here}}
    // expected-error-re@+5{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oma_local4' as '__single'}}
    // expected-note@+2{{pointer 'oma_local4' declared here}}
    // FIXIT-CHECK: MACRO_TY* __single oma_local4 = 0;
    MACRO_TY* oma_local4 = 0; // Fix
    oma_local4 = imp_single;
    oma_local4 =  single_source();
}

int block_local_fn(void* __single param) {
    int (^block_func)(void) = ^{
        // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'block_local' as '__single'}}
        // expected-note@+2{{pointer 'block_local' declared here}}
        // FIXIT-CHECK: void* __single block_local = param;
        void* block_local = param;

        // expected-error-re@+4{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'block_local2' as '__single'}}
        // expected-note@+2{{pointer 'block_local2' declared here}}
        // FIXIT-CHECK: void* __single block_local2;
        void* block_local2;
        block_local2 = param;

        return void_ptr_single_sink(block_local);
    };

    return block_func();
}

__ptrcheck_abi_assume_bidi_indexable()
typedef struct StructWithBidiPtrOIFD {
    // expected-note@+2{{pointer 'StructWithBidiPtrOIFD::oifd_bidi_ptr' declared here}}
    // FIXIT-CHECK: void* __single oifd_bidi_ptr;
    void* oifd_bidi_ptr;
} StructWithBidiPtrOIFD_t;

typedef struct StructWithBidiPtrOIFD2 {
    // expected-note@+2{{pointer 'StructWithBidiPtrOIFD2::oifd_bidi_ptr2' declared here}}
    // FIXIT-CHECK: void* __single oifd_bidi_ptr2;
    void* oifd_bidi_ptr2;
} StructWithBidiPtrOIFD2_t;

typedef struct StructWithBidiPtrOAFD {
    // expected-note@+2{{pointer 'StructWithBidiPtrOAFD::oafd_bidi_ptr' declared here}}
    // FIXIT-CHECK: void* __single oafd_bidi_ptr;
    void* oafd_bidi_ptr;
} StructWithBidiPtrOAFD_t;

typedef struct StructWithBidiPtrOAFD2 {
    // expected-note@+2{{pointer 'StructWithBidiPtrOAFD2::oafd_bidi_ptr2' declared here}}
    // FIXIT-CHECK: void* __single oafd_bidi_ptr2;
    void* oafd_bidi_ptr2;
} StructWithBidiPtrOAFD2_t;

typedef struct StructWithBidiPtrOAFD3 {
    // expected-note@+2{{pointer 'StructWithBidiPtrOAFD3::oafd_bidi_ptr3' declared here}}
    // FIXIT-CHECK: void* __single oafd_bidi_ptr3;
    void* oafd_bidi_ptr3;
} StructWithBidiPtrOAFD3_t;

typedef struct StructWithBidiPtrOAFD4 {
    // expected-note@+2{{pointer 'StructWithBidiPtrOAFD4::oafd_bidi_ptr4' declared here}}
    // FIXIT-CHECK: void* __single oafd_bidi_ptr4;
    void* oafd_bidi_ptr4;
} StructWithBidiPtrOAFD4_t;

typedef struct StructWithBidiPtrOMAFD {
    // expected-note@+3{{pointer 'StructWithBidiPtrOMAFD::omafd_bidi_ptr' declared here}}
    // expected-note@+2{{pointer 'StructWithBidiPtrOMAFD::omafd_bidi_ptr' declared here}}
    // FIXIT-CHECK: void* __single omafd_bidi_ptr;
    void* omafd_bidi_ptr;
} StructWithBidiPtrOMAFD_t;



__ptrcheck_abi_assume_indexable();

typedef struct StructWithIdxPtrOIFD {
    // expected-note@+2{{pointer 'StructWithIdxPtrOIFD::oifd_idx_ptr' declared here}}
    // FIXIT-CHECK: void* __single oifd_idx_ptr;
    void* oifd_idx_ptr;
} StructWithIdxPtrOIFD_t;

typedef struct StructWithIdxPtrOIFD2 {
    // expected-note@+2{{pointer 'StructWithIdxPtrOIFD2::oifd_idx_ptr2' declared here}}
    // FIXIT-CHECK: void* __single oifd_idx_ptr2;
    void* oifd_idx_ptr2;
} StructWithIdxPtrOIFD2_t;

typedef struct StructWithIdxPtrOAFD {
    // expected-note@+2{{pointer 'StructWithIdxPtrOAFD::oafd_idx_ptr' declared here}}
    // FIXIT-CHECK: void* __single oafd_idx_ptr;
    void* oafd_idx_ptr;
} StructWithIdxPtrOAFD_t;

typedef struct StructWithIdxPtrOAFD2 {
    // expected-note@+2{{pointer 'StructWithIdxPtrOAFD2::oafd_idx_ptr2' declared here}}
    // FIXIT-CHECK: void* __single oafd_idx_ptr2;
    void* oafd_idx_ptr2;
} StructWithIdxPtrOAFD2_t;

typedef struct StructWithIdxPtrOAFD3 {
    // expected-note@+2{{pointer 'StructWithIdxPtrOAFD3::oafd_idx_ptr3' declared here}}
    // FIXIT-CHECK: void* __single oafd_idx_ptr3;
    void* oafd_idx_ptr3;
} StructWithIdxPtrOAFD3_t;

typedef struct StructWithIdxPtrOAFD4 {
    // expected-note@+2{{pointer 'StructWithIdxPtrOAFD4::oafd_idx_ptr4' declared here}}
    // FIXIT-CHECK: void* __single oafd_idx_ptr4;
    void* oafd_idx_ptr4;
} StructWithIdxPtrOAFD4_t;

typedef struct StructWithIdxPtrOMAFD {
    // expected-note@+2 2 {{pointer 'StructWithIdxPtrOMAFD::omafd_idx_ptr' declared here}}
    // FIXIT-CHECK: void* __single omafd_idx_ptr;
    void* omafd_idx_ptr;
} StructWithIdxPtrOMAFD_t;


__ptrcheck_abi_assume_single()

typedef struct Nested_Bidi_OIFD {
    StructWithBidiPtrOIFD2_t field;
} Nested_Bidi_OIFD_t;

typedef struct Nested_Idx_OIFD {
    StructWithIdxPtrOIFD2_t field;
} Nested_Idx_OIFD_t;

void opaque_init_field_decl(void* __single explicit_single) {

    // expected-error-re@+1{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithBidiPtrOIFD::oifd_bidi_ptr' as '__single'}}
    StructWithBidiPtrOIFD_t oifd_local = {.oifd_bidi_ptr = explicit_single };

    // expected-error-re@+1{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithIdxPtrOIFD::oifd_idx_ptr' as '__single'}}
    StructWithIdxPtrOIFD_t oifd_local2 = {.oifd_idx_ptr = explicit_single };

    // expected-error-re@+1{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithBidiPtrOIFD2::oifd_bidi_ptr2' as '__single'}}
    Nested_Bidi_OIFD_t oifd_local3 = {.field = {.oifd_bidi_ptr2 = explicit_single}};

    // expected-error-re@+1{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithIdxPtrOIFD2::oifd_idx_ptr2' as '__single'}}
    Nested_Idx_OIFD_t oifd_local4 = {. field = {.oifd_idx_ptr2 = explicit_single}};
}

typedef struct Nested_Bidi_OAFD {
    StructWithBidiPtrOAFD3_t field;
} Nested_Bidi_OAFD_t;

typedef struct Nested_Idx_OAFD {
    StructWithIdxPtrOAFD3_t field;
} Nested_Idx_OAFD_t;

void opaque_assign_field_decl(void* __single explicit_single, StructWithBidiPtrOAFD2_t* base_with_bidi_field, StructWithIdxPtrOAFD2_t* base_with_idx_field) {
    StructWithBidiPtrOAFD_t oafd_local;
    StructWithIdxPtrOAFD_t oafd_local2;
    
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithBidiPtrOAFD::oafd_bidi_ptr' as '__single'}}
    oafd_local.oafd_bidi_ptr = explicit_single;
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithIdxPtrOAFD::oafd_idx_ptr' as '__single'}}
    oafd_local2.oafd_idx_ptr = explicit_single;

    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithBidiPtrOAFD2::oafd_bidi_ptr2' as '__single'}}
    base_with_bidi_field->oafd_bidi_ptr2 = explicit_single;
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithIdxPtrOAFD2::oafd_idx_ptr2' as '__single'}}
    base_with_idx_field->oafd_idx_ptr2 = explicit_single;

    Nested_Bidi_OAFD_t oafd_local3;
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithBidiPtrOAFD3::oafd_bidi_ptr3' as '__single'}}
    oafd_local3.field.oafd_bidi_ptr3 = explicit_single;

    Nested_Idx_OAFD_t oafd_local4;
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithIdxPtrOAFD3::oafd_idx_ptr3' as '__single'}}
    oafd_local4.field.oafd_idx_ptr3 = explicit_single;

    // Test assignment with parentheses on LHS
    StructWithBidiPtrOAFD4_t oafd_local5;
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithBidiPtrOAFD4::oafd_bidi_ptr4' as '__single'}}
    (oafd_local5.oafd_bidi_ptr4) = explicit_single;

        // Test assignment with parentheses on LHS
    StructWithIdxPtrOAFD4_t oafd_local6;
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithIdxPtrOAFD4::oafd_idx_ptr4' as '__single'}}
    (oafd_local6.oafd_idx_ptr4) = explicit_single;
}

void opaque_multiple_assign_field_decl(void* __single explicit_single) {
    StructWithBidiPtrOMAFD_t omafd_local;
    StructWithIdxPtrOMAFD_t omafd_local2;

    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithBidiPtrOMAFD::omafd_bidi_ptr' as '__single'}}
    omafd_local.omafd_bidi_ptr = explicit_single;
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithBidiPtrOMAFD::omafd_bidi_ptr' as '__single'}}
    omafd_local.omafd_bidi_ptr = explicit_single;

    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithIdxPtrOMAFD::omafd_idx_ptr' as '__single'}}
    omafd_local2.omafd_idx_ptr = explicit_single;
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithIdxPtrOMAFD::omafd_idx_ptr' as '__single'}}
    omafd_local2.omafd_idx_ptr = explicit_single;
}


// This defined to be a compile time constant because it seems this is the only
// thing clang with -fbounds-safety allows.
#define SINGLE_SOURCE_GLOBAL ((F_t* __single) 0)

// Assigning single_source() isn't legal here because it isn't a constant.
// FIXIT-CHECK: F_t* global_F_single = SINGLE_SOURCE_GLOBAL;
F_t* global_F_single = SINGLE_SOURCE_GLOBAL; // No Fix

__ptrcheck_abi_assume_bidi_indexable();

// It's questionable if the FixIts for these global assignments are of value.
// Once the fix is made compilation will fail because `__single_source()` isn't
// a compile time constant.

// expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_bidi' as '__single'}}
// expected-note@+2{{pointer 'global_F_implicit_bidi' declared here}}
// FIXIT-CHECK: F_t* __single global_F_implicit_bidi = single_source();
F_t* global_F_implicit_bidi = single_source(); // Fix

// expected-note@+2{{pointer 'global_F_implicit_bidi2' declared here}}
// FIXIT-CHECK: F_t* __single global_F_implicit_bidi2;
F_t* global_F_implicit_bidi2;
void modify_global_implicit_bidi(F_t* __single explicit_single) {
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_bidi2' as '__single'}}
    global_F_implicit_bidi2 = explicit_single;
}


// FIXIT-CHECK: F_t* __single global_F_implicit_bidi3;
// FIXIT-CHECK-NEXT: F_t* __single global_F_implicit_bidi3;
// FIXIT-CHECK-NEXT: F_t* __single global_F_implicit_bidi3;
// expected-note@+3{{pointer 'global_F_implicit_bidi3' declared here}}
F_t* global_F_implicit_bidi3; // Fix
F_t* global_F_implicit_bidi3; // Fix
F_t* global_F_implicit_bidi3; // Fix
void modify_global_implicit_bidi2(F_t* __single explicit_single) {
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_bidi3' as '__single'}}
    global_F_implicit_bidi3 = explicit_single;
}

// expected-note@+2 3 {{pointer 'global_F_implicit_bidi4' declared here}}
// FIXIT-CHECK: F_t* __single global_F_implicit_bidi4;
F_t* global_F_implicit_bidi4; // Fix
void modify_global_implicit_bidi_multiple_assign(F_t* __single explicit_single) {
    // expected-error-re@+3{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_bidi4' as '__single'}}
    // expected-error-re@+3{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_bidi4' as '__single'}}
    // expected-error-re@+3{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_bidi4' as '__single'}}
    global_F_implicit_bidi4 = explicit_single;
    global_F_implicit_bidi4 = explicit_single;
    global_F_implicit_bidi4 = explicit_single;
}

// This tests the case where a global gets a FixIt emitted on it and then it
// gets redeclared and then a FixIt gets emitted on the redeclaration. We have
// to be careful to not annotate the first declaration again.

// expected-note@+2{{pointer 'global_F_implicit_bidi5' declared here}}
// FIXIT-CHECK: F_t* __single global_F_implicit_bidi5;
F_t* global_F_implicit_bidi5; // First Decl, Fix

void modify_global_implicit_bidi_pre_redeclare(F_t* __single explicit_single) {
    // Fixes First Decl
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_bidi5' as '__single'}}
    global_F_implicit_bidi5 = explicit_single;
}

// expected-note@+2{{pointer 'global_F_implicit_bidi5' declared here}}
// FIXIT-CHECK: F_t* __single global_F_implicit_bidi5;
F_t* global_F_implicit_bidi5; // Second Decl, Fix

void modify_global_implicit_bidi_post_redeclare(F_t* __single explicit_single) {
    // Fixes Second Decl. Have to avoid fixing First Decl
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_bidi5' as '__single'}}
    global_F_implicit_bidi5 = explicit_single;
}

// FIXME(dliew): rdar://115456779
// This is a case where not all the redeclarations can be fixed
// expected-note@+2{{pointer 'global_F_implicit_bidi6' declared here}}
// FIXIT-CHECK: F_t* __single global_F_implicit_bidi6;
F_t* global_F_implicit_bidi6; // First Decl, Fix

void modify_global_implicit_bidi_pre_redeclare2(F_t* __single explicit_single) {
    // Fixes Second Decl. Have to avoid fixing First Decl
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_bidi6' as '__single'}}
    global_F_implicit_bidi6 = explicit_single;
}

// No note
// FIXIT-CHECK: F_t* global_F_implicit_bidi6;
F_t* global_F_implicit_bidi6; // Second Decl, No Fix

__ptrcheck_abi_assume_indexable();

// expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_idx' as '__single'}}
// expected-note@+2{{pointer 'global_F_implicit_idx' declared here}}
// FIXIT-CHECK: F_t* __single global_F_implicit_idx = single_source();
F_t* global_F_implicit_idx = single_source(); // Fix

// expected-note@+2{{pointer 'global_F_implicit_idx2' declared here}}
// FIXIT-CHECK: F_t* __single global_F_implicit_idx2;
F_t* global_F_implicit_idx2;
void modify_global_implicit_idx(F_t* __single explicit_single) {
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_idx2' as '__single'}}
    global_F_implicit_idx2 = explicit_single;
}

// FIXIT-CHECK: F_t* __single global_F_implicit_idx3;
// FIXIT-CHECK-NEXT: F_t* __single global_F_implicit_idx3;
// FIXIT-CHECK-NEXT: F_t* __single global_F_implicit_idx3;
// expected-note@+3{{pointer 'global_F_implicit_idx3' declared here}}
F_t* global_F_implicit_idx3; // Fix
F_t* global_F_implicit_idx3; // Fix
F_t* global_F_implicit_idx3; // Fix
void modify_global_implicit_idx2(F_t* __single explicit_single) {
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_idx3' as '__single'}}
    global_F_implicit_idx3 = explicit_single;
}

// expected-note@+2 3 {{pointer 'global_F_implicit_idx4' declared here}}
// FIXIT-CHECK: F_t* __single global_F_implicit_idx4;
F_t* global_F_implicit_idx4; // Fix
void modify_global_implicit_idx_multiple_assign(F_t* __single explicit_single) {
    // expected-error-re@+3{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_idx4' as '__single'}}
    // expected-error-re@+3{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_idx4' as '__single'}}
    // expected-error-re@+3{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_idx4' as '__single'}}
    global_F_implicit_idx4 = explicit_single;
    global_F_implicit_idx4 = explicit_single;
    global_F_implicit_idx4 = explicit_single;
}

// This tests the case where a global gets a FixIt emitted on it and then it
// gets redeclared and then a FixIt gets emitted on the redeclaration. We have
// to be careful to not annotate the first declaration again.

// expected-note@+2{{pointer 'global_F_implicit_idx5' declared here}}
// FIXIT-CHECK: F_t* __single global_F_implicit_idx5;
F_t* global_F_implicit_idx5; // First Decl, Fix

void modify_global_implicit_idx_pre_redeclare(F_t* __single explicit_single) {
    // Fixes First Decl
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_idx5' as '__single'}}
    global_F_implicit_idx5 = explicit_single;
}

// expected-note@+2{{pointer 'global_F_implicit_idx5' declared here}}
// FIXIT-CHECK: F_t* __single global_F_implicit_idx5;
F_t* global_F_implicit_idx5; // Second Decl, Fix

void modify_global_implicit_idx_post_redeclare(F_t* __single explicit_single) {
    // Fixes Second Decl. Have to avoid fixing First Decl
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_idx5' as '__single'}}
    global_F_implicit_idx5 = explicit_single;
}

// FIXME(dliew): rdar://115456779
// This is a case where not all the redeclarations can be fixed
// expected-note@+2{{pointer 'global_F_implicit_idx6' declared here}}
// FIXIT-CHECK: F_t* __single global_F_implicit_idx6;
F_t* global_F_implicit_idx6; // First Decl, Fix

void modify_global_implicit_idx_pre_redeclare2(F_t* __single explicit_single) {
    // Fixes Second Decl. Have to avoid fixing First Decl
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_implicit_idx6' as '__single'}}
    global_F_implicit_idx6 = explicit_single;
}

// No note
// FIXIT-CHECK: F_t* global_F_implicit_idx6;
F_t* global_F_implicit_idx6; // Second Decl, No Fix


__ptrcheck_abi_assume_single();

//==============================================================================
// FixIts on parameters
//==============================================================================

__ptrcheck_abi_assume_bidi_indexable();

// expected-note@+2{{passing argument to parameter 'implicit_bidi' here}}
// FIXIT-CHECK: void implicit_bidi_sink(void* __single implicit_bidi);
void implicit_bidi_sink(void* implicit_bidi); // Fix

// expected-note@+3{{pointer 'implicit_bidi2' declared here}} // This is emitted due to a FixIt needs to be attached to a diagnostic and we can't attach it to the error.
// expected-note@+2 2 {{passing argument to parameter 'implicit_bidi2' here}}
// FIXIT-CHECK: void implicit_bidi_sink2(void* __single implicit_bidi2);
void implicit_bidi_sink2(void* implicit_bidi2); // Fix

// expected-note@+3{{pointer declared here}} // This is emitted due to a FixIt needs to be attached to a diagnostic and we can't attach it to the error.
// expected-note@+2 2 {{passing argument to parameter here}}
// FIXIT-CHECK: void implicit_bidi_sink3(void*__single);
void implicit_bidi_sink3(void*); // Fix

__ptrcheck_abi_assume_single();

void single_opaque_arg_passed_to_implicit_bidi_param(void* __single p) {
    // expected-error@+1{{cannot pass __single pointer to incomplete type argument 'void *__single' when parameter is an indexable pointer 'void *__bidi_indexable'; consider making the 'implicit_bidi' parameter '__single'}}
    implicit_bidi_sink(p);
}

void single_opaque_arg_passed_to_implicit_bidi_param2(void* __single p) {
    // expected-error@+2{{cannot pass __single pointer to incomplete type argument 'void *__single' when parameter is an indexable pointer 'void *__bidi_indexable'; consider making the 'implicit_bidi2' parameter '__single'}}
    // expected-error@+2{{cannot pass __single pointer to incomplete type argument 'void *__single' when parameter is an indexable pointer 'void *__bidi_indexable'; consider making the 'implicit_bidi2' parameter '__single'}}
    implicit_bidi_sink2(p);
    implicit_bidi_sink2(p);
}

void single_opaque_arg_passed_to_implicit_bidi_param3(void* __single p) {
    // expected-error@+2{{cannot pass __single pointer to incomplete type argument 'void *__single' when parameter is an indexable pointer 'void *__bidi_indexable'; consider making the parameter '__single'}}
    // expected-error@+2{{cannot pass __single pointer to incomplete type argument 'void *__single' when parameter is an indexable pointer 'void *__bidi_indexable'; consider making the parameter '__single'}}
    implicit_bidi_sink3(p);
    implicit_bidi_sink3(p);
}

__ptrcheck_abi_assume_indexable();

// expected-note@+2{{passing argument to parameter 'implicit_idx_param' here}}
// FIXIT-CHECK: void implicit_idx_sink(void* __single implicit_idx_param);
void implicit_idx_sink(void* implicit_idx_param); // Fix

// expected-note@+3{{pointer 'implicit_idx_param2' declared here}} // This is emitted due to a FixIt needs to be attached to a diagnostic and we can't attach it to the error.
// expected-note@+2 2 {{passing argument to parameter 'implicit_idx_param2' here}}
// FIXIT-CHECK: void implicit_idx_sink2(void* __single implicit_idx_param2);
void implicit_idx_sink2(void* implicit_idx_param2); // Fix

// expected-note@+3{{pointer declared here}} // This is emitted due to a FixIt needs to be attached to a diagnostic and we can't attach it to the error.
// expected-note@+2 2 {{passing argument to parameter here}}
// FIXIT-CHECK: void implicit_idx_sink3(void*__single);
void implicit_idx_sink3(void*); // Fix

__ptrcheck_abi_assume_single();

void single_opaque_arg_passed_to_implicit_idx_param(void* __single p) {
    // expected-error@+1{{cannot pass __single pointer to incomplete type argument 'void *__single' when parameter is an indexable pointer 'void *__indexable'; consider making the 'implicit_idx_param' parameter '__single'}}
    implicit_idx_sink(p);
}

void single_opaque_arg_passed_to_implicit_idx_param2(void* __single p) {
    // expected-error@+2{{cannot pass __single pointer to incomplete type argument 'void *__single' when parameter is an indexable pointer 'void *__indexable'; consider making the 'implicit_idx_param2' parameter '__single'}}
    // expected-error@+2{{cannot pass __single pointer to incomplete type argument 'void *__single' when parameter is an indexable pointer 'void *__indexable'; consider making the 'implicit_idx_param2' parameter '__single'}}
    implicit_idx_sink2(p);
    implicit_idx_sink2(p);
}

void single_opaque_arg_passed_to_implicit_idx_param3(void* __single p) {
    // expected-error@+2{{cannot pass __single pointer to incomplete type argument 'void *__single' when parameter is an indexable pointer 'void *__indexable'; consider making the parameter '__single'}}
    // expected-error@+2{{cannot pass __single pointer to incomplete type argument 'void *__single' when parameter is an indexable pointer 'void *__indexable'; consider making the parameter '__single'}}
    implicit_idx_sink3(p);
    implicit_idx_sink3(p);
}

//==============================================================================
// No FixIts for array assignments/initialization
//
// The lack of FixIts here is just an implementation limitation.
// rdar://115201001
//==============================================================================

__ptrcheck_abi_assume_bidi_indexable();

// FIXIT-CHECK: void* global_opaque_array_with_init[] = { single_source(), 0};
// expected-error-re@+1{{cannot initialize indexable pointer with type 'void *__bidi_indexable' from __single pointer to incomplete type 'F_t *__single' (aka 'struct F *__single'){{$}}}}
void* global_opaque_array_with_init[] = { single_source(), 0}; // No Fix

// FIXIT-CHECK: void* global_opaque_array[4];
void* global_opaque_array[4]; // No Fix
__ptrcheck_abi_assume_single();

void assign_global_opaque_array(void* __single explicit_single) {
    // The suggestion to `consider` making the destination __single here is omitted
    // expected-error-re@+1{{cannot assign to indexable pointer with type 'void *__bidi_indexable' from __single pointer to incomplete type 'void *__single'{{$}}}}
    global_opaque_array[0] = explicit_single;
}

void assign_local_opaque_array(void* __single explicit_single) {
    void* aloa_array[2]; // Implicitly void* __single aloa_array[2];
    aloa_array[0] = explicit_single; // No diagnostic
}

//==============================================================================
// No FixIts
//
// The lack of FixIts here is just an implementation limitation.
// rdar://114478465
//==============================================================================

typedef struct StructWithExplicitBidiPtrOIAEB {
    // expected-note@+2{{pointer 'StructWithExplicitBidiPtrOIAEB::oiaeb_bidi_ptr' declared here}}
    // FIXIT-CHECK: void* __bidi_indexable oiaeb_bidi_ptr;
    void* __bidi_indexable oiaeb_bidi_ptr; // No fix
} StructWithExplicitBidiPtrOIAEB_t;

typedef struct StructWithExplicitIdxPtrOIAEB {
    // expected-note@+2{{pointer 'StructWithExplicitIdxPtrOIAEB::oiaeb_idx_ptr' declared here}}
    // FIXIT-CHECK: void* __indexable oiaeb_idx_ptr;
    void* __indexable oiaeb_idx_ptr; // No fix
} StructWithExplicitIdxPtrOIAEB_t;

int opaque_init_assign_explicit_bidi(F_t* imp_single) {
    // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oiaeb_local1' as '__single'}}
    // expected-note@+2{{pointer 'oiaeb_local1' declared here}}
    // FIXIT-CHECK: F_t* __bidi_indexable oiaeb_local1 = imp_single;
    F_t* __bidi_indexable oiaeb_local1 = imp_single; // No fix

    // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oiaeb_local2' as '__single'}}
    // expected-note@+2{{pointer 'oiaeb_local2' declared here}}
    // FIXIT-CHECK: F_t* __bidi_indexable oiaeb_local2 = single_source();
    F_t* __bidi_indexable oiaeb_local2 = single_source(); // No fix

    // expected-error-re@+1{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithExplicitBidiPtrOIAEB::oiaeb_bidi_ptr' as '__single'}}
    StructWithExplicitBidiPtrOIAEB_t oiaeb_local3 = {.oiaeb_bidi_ptr = imp_single};

    // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oiaeb_local4' as '__single'}}
    // expected-note@+2{{pointer 'oiaeb_local4' declared here}}
    // FIXIT-CHECK: MACRO_PTR_TY __bidi_indexable oiaeb_local4 = imp_single;
    MACRO_PTR_TY __bidi_indexable oiaeb_local4 = imp_single; // No Fix

    // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oiaeb_local5' as '__single'}}
    // expected-note@+2{{pointer 'oiaeb_local5' declared here}}
    // FIXIT-CHECK: MACRO_TY* __bidi_indexable oiaeb_local5 = imp_single;
    MACRO_TY* __bidi_indexable oiaeb_local5 = imp_single; // No Fix
}

typedef struct StructWithExplicitBidiPtrOAEB {
    // expected-note@+2{{pointer 'StructWithExplicitBidiPtrOAEB::oaeb_bidi_ptr' declared here}}
    // FIXIT-CHECK: void* __bidi_indexable oaeb_bidi_ptr;
    void* __bidi_indexable oaeb_bidi_ptr; // No fix
} StructWithExplicitBidiPtrOAEB_t;

typedef struct StructWithExplicitIdxPtrOAEI {
    // expected-note@+2{{pointer 'StructWithExplicitIdxPtrOAEI::oaei_idx_ptr' declared here}}
    // FIXIT-CHECK: void* __indexable oaei_idx_ptr;
    void* __indexable oaei_idx_ptr; // No fix
} StructWithExplicitIdxPtrOAEI_t;

int opaque_assign_explicit_bidi(F_t* imp_single) {
    // expected-error-re@+4{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oaeb_local1' as '__single'}}
    // expected-note@+2{{pointer 'oaeb_local1' declared here}}
    // FIXIT-CHECK: F_t* __bidi_indexable oaeb_local1 = 0;
    F_t* __bidi_indexable oaeb_local1 = 0; // No Fix
    oaeb_local1 = imp_single;

    // expected-error-re@+4{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oaeb_local2' as '__single'}}
    // expected-note@+2{{pointer 'oaeb_local2' declared here}}
    // FIXIT-CHECK: F_t* __bidi_indexable oaeb_local2 = 0;
    F_t* __bidi_indexable oaeb_local2 = 0; // No Fix
    oaeb_local2 = single_source();

    StructWithExplicitBidiPtrOAEB_t oaeb_local3;
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithExplicitBidiPtrOAEB::oaeb_bidi_ptr' as '__single'}}
    oaeb_local3.oaeb_bidi_ptr = imp_single;
}

int opaque_init_assign_explicit_idx(F_t* imp_single) {
    // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oiaei_local1' as '__single'}}
    // expected-note@+2{{pointer 'oiaei_local1' declared here}}
    // FIXIT-CHECK: F_t* __indexable oiaei_local1 = imp_single;
    F_t* __indexable oiaei_local1 = imp_single; // No fix

    // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oiaei_local2' as '__single'}}
    // expected-note@+2{{pointer 'oiaei_local2' declared here}}
    // FIXIT-CHECK: F_t* __indexable oiaei_local2 = single_source();
    F_t* __indexable oiaei_local2 = single_source(); // No fix

    // expected-error-re@+1{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithExplicitIdxPtrOIAEB::oiaeb_idx_ptr' as '__single'}}
    StructWithExplicitIdxPtrOIAEB_t oiaei_local3 = {.oiaeb_idx_ptr = imp_single};

    // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oiaei_local4' as '__single'}}
    // expected-note@+2{{pointer 'oiaei_local4' declared here}}
    // FIXIT-CHECK: MACRO_PTR_TY __indexable oiaei_local4 = imp_single;
    MACRO_PTR_TY __indexable oiaei_local4 = imp_single; // No Fix

    // expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oiaei_local5' as '__single'}}
    // expected-note@+2{{pointer 'oiaei_local5' declared here}}
    // FIXIT-CHECK: MACRO_TY* __indexable oiaei_local5 = imp_single
    MACRO_TY* __indexable oiaei_local5 = imp_single; // No Fix
}

int opaque_assign_explicit_idx(F_t* imp_single) {
    // expected-error-re@+4{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oaei_local1' as '__single'}}
    // expected-note@+2{{pointer 'oaei_local1' declared here}}
    // FIXIT-CHECK: F_t* __indexable oaei_local1 = 0;
    F_t* __indexable oaei_local1 = 0; // No Fix
    oaei_local1 = imp_single;

    // expected-error-re@+4{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'oaei_local2' as '__single'}}
    // expected-note@+2{{pointer 'oaei_local2' declared here}}
    // FIXIT-CHECK: F_t* __indexable oaei_local2 = 0;
    F_t* __indexable oaei_local2 = 0; // No Fix
    oaei_local2 = single_source();

    StructWithExplicitIdxPtrOAEI_t oaei_local3;
    // expected-error-re@+1{{cannot assign to indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'StructWithExplicitIdxPtrOAEI::oaei_idx_ptr' as '__single'}}
    oaei_local3.oaei_idx_ptr = imp_single;
}

// expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_explicit_bidi' as '__single'}}
// expected-note@+2{{pointer 'global_F_explicit_bidi' declared here}}
// FIXIT-CHECK: F_t* __bidi_indexable global_F_explicit_bidi = single_source();
F_t* __bidi_indexable global_F_explicit_bidi = single_source(); // No Fix

// expected-error-re@+3{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type {{.+}}; consider declaring pointer 'global_F_explicit_idx' as '__single'}}
// expected-note@+2{{pointer 'global_F_explicit_idx' declared here}}
// FIXIT-CHECK: F_t* __indexable global_F_explicit_idx = single_source();
F_t* __indexable global_F_explicit_idx = single_source(); // No Fix

//==============================================================================
// No FixIts
//==============================================================================

// This is cast so there's no destination variable to modify at the point the
// diagnostic is emitted.
F_t* foo3(F_t* imp_single) {
    // expected-error-re@+2{{cannot cast from __single pointer to incomplete type {{.+}} to indexable pointer type {{.+}}}}
    // FIXIT-CHECK: return (F_t* __bidi_indexable) imp_single;
    return (F_t* __bidi_indexable) imp_single; // No Fix
}
