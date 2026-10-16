// REQUIRES: x86-registered-target
// RUN: %clang_cl -c --target=x86_64-windows-msvc -EHs-c- -O2 -GS- \
// RUN:   -Xclang=-import-call-optimization \
// RUN:   -clang:-S -clang:-o- -- %s 2>&1 \
// RUN:   | FileCheck %s

#ifdef __clang__
#define NO_TAIL __attribute((disable_tail_calls))
#else
#define NO_TAIL
#endif

void might_throw();
void other_func(int x);

void does_not_throw() noexcept(true);

extern "C" void __declspec(dllimport) some_dll_import();

class HasDtor {
    int x;
    char foo[40];

public:
    explicit HasDtor(int x);
    ~HasDtor();
};

void normal_has_regions() {
    {
        HasDtor hd{42};

        // because state changes, we expect the HasDtor::HasDtor() call to have a NOP
        might_throw();
    }

    other_func(10);
}
// CHECK-LABEL: .def "?normal_has_regions@@YAXXZ"
// CHECK: .seh_endprologue
// CHECK: call "??0HasDtor@@QEAA@H@Z"
// CHECK-NEXT: call "?might_throw@@YAXXZ"
// CHECK-NEXT: mov
// CHECK: call "??1HasDtor@@QEAA@XZ"
// CHECK-NEXT: mov ecx, 10
// CHECK-NEXT: call "?other_func@@YAXH@Z"
// CHECK-NEXT: nop
// CHECK-NEXT: .seh_startepilogue
// CHECK-NOT: "$ip2state$?normal_has_regions@@YAXXZ"

// This tests a tail call to a destructor.
void case_dtor_arg_empty_body(HasDtor x)
{
}
// CHECK-LABEL: .def "?case_dtor_arg_empty_body@@YAXVHasDtor@@@Z"
// CHECK: jmp "??1HasDtor@@QEAA@XZ"

int case_dtor_arg_empty_with_ret(HasDtor x)
{
    // The call to HasDtor::~HasDtor() should NOT have a NOP because the
    // following "mov eax, 100" instruction is in the same EH state.
    return 100;
}
// CHECK-LABEL: .def "?case_dtor_arg_empty_with_ret@@YAHVHasDtor@@@Z"
// CHECK: .seh_endprologue
// CHECK: call "??1HasDtor@@QEAA@XZ"
// CHECK-NOT: nop
// CHECK: mov eax, 100
// CHECK: .seh_startepilogue
// CHECK: .seh_endepilogue
// CHECK: .seh_endproc

void case_except_simple_call() NO_TAIL
{
    does_not_throw();
}

// This tests that the destructor is called right before SEH_BeginEpilogue,
// but in a function that has a return value.
int case_dtor_arg_calls_no_throw(HasDtor x)
{
    does_not_throw(); // no NOP expected
    return 100;
}

// Check the behavior of CALLs that are at the end of MBBs. If a CALL is within
// a non-null EH state (state -1) and is at the end of an MBB, then we expect
// to find an EH_LABEL after the CALL. This causes us to insert a NOP, which
// is the desired result.
void case_dtor_runs_after_join(int x) {

    // ctor call does not need a NOP, because it has real instructions after it
    HasDtor hd{42};

    if (x) {
        might_throw();
    } else {
        other_func(10);
    }
    does_not_throw();
    // ~HasDtor() runs
}

// CHECK-LABEL: .def "?case_dtor_runs_after_join@@YAXH@Z"
// CHECK: .seh_endprologue
// CHECK: call "??0HasDtor@@QEAA@H@Z"
// CHECK-NEXT: test
// CHECK: call "?might_throw@@YAXXZ"
// CHECK-NEXT: jmp
// CHECK: call "?other_func@@YAXH@Z"
// CHECK-NEXT: .LBB
// CHECK: call "?does_not_throw@@YAXXZ"
// CHECK-NEXT: lea
// CHECK-NEXT: call "??1HasDtor@@QEAA@XZ"
// CHECK-NEXT: nop
// CHECK-NEXT: .seh_startepilogue
// CHECK-NOT: "$ip2state$?case_dtor_runs_after_join@@YAXH@Z":


// Check the behavior of NOP padding around tail calls.
// We do not expect to insert NOPs around tail calls.
// However, the first call (to other_func()) does get a NOP
// because it comes before .seh_startepilogue.
void case_tail_call_no_eh() {
    // ordinary call
    other_func(10);

    // tail call; no NOP padding after JMP
    does_not_throw();
}

// CHECK-LABEL: .def "?case_tail_call_no_eh@@YAXXZ"
// CHECK: .seh_endprologue
// CHECK: call "?other_func@@YAXH@Z"
// CHECK-NEXT: nop
// CHECK-NEXT: .seh_startepilogue
// CHECK: .seh_endepilogue
// CHECK: jmp "?does_not_throw@@YAXXZ"
// CHECK-NOT: nop
// CHECK: .seh_endproc
