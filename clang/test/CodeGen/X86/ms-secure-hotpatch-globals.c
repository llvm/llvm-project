// REQUIRES: x86-registered-target

// This verifies that global variable redirection works correctly when using hotpatching.
//
// RUN: %clang_cl -c --target=x86_64-windows-msvc -O2 /Z7 \
// RUN:   -fms-secure-hotpatch-functions-list=hp1,hp2,hp3,hp4,hp5_phi_ptr_mixed,hp_phi_ptr_both,hp_const_ptr_sub \
// RUN:   /clang:-S /clang:-o- -- %s | FileCheck %s

#ifdef __clang__
#define NO_TAIL __attribute__((disable_tail_calls))
#else
#define NO_TAIL
#endif

extern int g_data[10];

struct SomeData {
    int x;
    int y;
};

const struct SomeData g_this_is_const = { 100, 200 };

struct HasPointers {
    int* ptr;
    int x;
};

extern struct HasPointers g_has_pointers;

void take_data(const void* p);

void do_side_effects();
void do_other_side_effects();

void hp1() NO_TAIL {
    take_data(&g_data[5]);
}

// CHECK: hp1:
// CHECK: mov rcx, qword ptr [rip + __ref_g_data]
// CHECK: add rcx, 20
// CHECK: call take_data
// CHECK: .seh_endproc

void hp2() NO_TAIL {
    // We do not expect string literals to be redirected.
    take_data("hello, world!");
}

// CHECK: hp2:
// CHECK: lea rcx, [rip + "??_C@_0O@KJBLMJCB@hello?0?5world?$CB?$AA@"]
// CHECK: call take_data
// CHECK: .seh_endproc

void hp3() NO_TAIL {
    // We do not expect g_this_is_const to be redirected because it is const
    // and contains no pointers.
    take_data(&g_this_is_const);
}

// CHECK: hp3:
// CHECK: lea rcx, [rip + g_this_is_const]
// CHECK: call take_data
// CHECK-NOT: __ref_g_this_is_const
// CHECK: .seh_endproc

void hp4() NO_TAIL {
    take_data(&g_has_pointers);
    // We expect &g_has_pointers to be redirected.
}

// CHECK: hp4:
// CHECK: mov rcx, qword ptr [rip + __ref_g_has_pointers]
// CHECK: call take_data
// CHECK: .seh_endproc

// This case checks that global variable redirection interacts correctly with PHI nodes.
// The IR for this generates a "phi ptr g_has_pointers, g_this_is_const" node.
// We expect g_has_pointers to be redirected, but not g_this_is_const.
void hp5_phi_ptr_mixed(int x) NO_TAIL {
    const void* y;
    if (x) {
        y = &g_has_pointers;
        do_side_effects();
    } else {
        y = &g_this_is_const;
        do_other_side_effects();
    }
    take_data(y);
}

// CHECK: hp5_phi_ptr_mixed
// CHECK: .seh_endprologue
// CHECK: test ecx, ecx
// CHECK: mov rsi, qword ptr [rip + __ref_g_has_pointers]
// CHECK: call do_side_effects
// CHECK: jmp
// CHECK: call do_other_side_effects
// CHECK: lea rsi, [rip + g_this_is_const]
// CHECK: mov rcx, rsi
// CHECK: call take_data
// CHECK: .seh_endproc

// This case tests that global variable redirection interacts correctly with PHI nodes,
// where two (all) operands of a given PHI node are globabl variables that redirect.
void hp_phi_ptr_both(int x) NO_TAIL {
    const void* y;
    if (x) {
        y = &g_has_pointers;
        do_side_effects();
    } else {
        y = &g_data[5];
        do_other_side_effects();
    }
    take_data(y);
}

// CHECK: hp_phi_ptr_both:
// CHECK: .seh_endprologue
// CHECK: test ecx, ecx
// CHECK: mov rsi, qword ptr [rip + __ref_g_has_pointers]
// CHECK: mov rsi, qword ptr [rip + __ref_g_data]
// CHECK: take_data
// CHECK: .seh_endproc

// Test a constant expression which references global variable addresses.
size_t hp_const_ptr_sub() NO_TAIL {
    return (unsigned char*)&g_has_pointers - (unsigned char*)&g_data;
}

// CHECK: hp_const_ptr_sub:
// CHECK: mov rax, qword ptr [rip + __ref_g_has_pointers]
// CHECK: sub rax, qword ptr [rip + __ref_g_data]
// CHECK: ret
