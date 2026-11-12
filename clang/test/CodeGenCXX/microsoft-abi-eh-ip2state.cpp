// REQUIRES: x86-registered-target
// RUN: %clang_cl -c --target=x86_64-windows-msvc -O2 -EHsc -GS- \
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

class BadError {
public:
    int errorCode;
};

// Verify that when NOP padding for IP2State is active *and* Import Call
// Optimization is active that we see both forms of NOP padding.
void case_calls_dll_import() NO_TAIL {
    some_dll_import();
}
// CHECK-LABEL: .def "?case_calls_dll_import@@YAXXZ"
// CHECK: .seh_endprologue
// CHECK: .Limpcall{{[0-9]+}}:
// CHECK-NEXT: rex64
// CHECK-NEXT: call __imp_some_dll_import
// CHECK-NEXT: nop dword ptr {{\[.*\]}}
// CHECK-NEXT: nop
// CHECK-NEXT: .seh_startepilogue

void normal_has_regions() {

    // <-- state -1 (none)
    {
        HasDtor hd{42};

        // <-- state goes from -1 to 0
        // because state changes, we expect the HasDtor::HasDtor() call to have a NOP

        might_throw();

        // <-- state goes from 0 to -1 because we're about to call HasDtor::~HasDtor()
        // <-- state -1
    }

    // <-- state -1
    other_func(10);

    // <-- state -1
}
// CHECK-LABEL: .def "?normal_has_regions@@YAXXZ"
// CHECK: .seh_endprologue
// CHECK: call "??0HasDtor@@QEAA@H@Z"
// CHECK-NEXT: nop
// CHECK: call "?might_throw@@YAXXZ"
// CHECK-NEXT: nop
// CHECK: call "??1HasDtor@@QEAA@XZ"
// CHECK: call "?other_func@@YAXH@Z"
// CHECK-NEXT: nop
// CHECK: .seh_startepilogue

// This tests a tail call to a destructor.
void case_dtor_arg_empty_body(HasDtor x)
{
}
// CHECK-LABEL: .def "?case_dtor_arg_empty_body@@YAXVHasDtor@@@Z"
// CHECK: jmp "??1HasDtor@@QEAA@XZ"

int case_dtor_arg_empty_with_ret(HasDtor x)
{
    // CHECK-LABEL: .def "?case_dtor_arg_empty_with_ret@@YAHVHasDtor@@@Z"
    // CHECK: .seh_endprologue

    // CHECK: call "??1HasDtor@@QEAA@XZ"
    // CHECK-NOT: nop

    // The call to HasDtor::~HasDtor() should NOT have a NOP because the
    // following "mov eax, 100" instruction is in the same EH state.

    return 100;

    // CHECK: mov eax, 100
    // CHECK: .seh_startepilogue
    // CHECK: .seh_endepilogue
    // CHECK: .seh_endproc
}

int case_noexcept_dtor(HasDtor x) noexcept(true)
{
    // CHECK: .def "?case_noexcept_dtor@@YAHVHasDtor@@@Z"
    // CHECK: call "??1HasDtor@@QEAA@XZ"
    // CHECK-NEXT: mov eax, 100
    // CHECK-NEXT: .seh_startepilogue
    return 100;
}

// Simple call of a function that can throw
void case_except_simple_call() NO_TAIL
{
    might_throw();
}
// CHECK-LABEL: .def "?case_except_simple_call@@YAXXZ"
// CHECK: .seh_endprologue
// CHECK-NEXT: call "?might_throw@@YAXXZ"
// CHECK-NEXT: nop
// CHECK-NEXT: .seh_startepilogue

// Simple call of a function that cannot throw, in a noexcept context.
void case_noexcept_simple_call() noexcept(true) NO_TAIL
{
    does_not_throw();
}
// CHECK-LABEL: .def "?case_noexcept_simple_call@@YAXXZ"
// CHECK: .seh_endprologue
// CHECK-NEXT: call "?does_not_throw@@YAXXZ"
// CHECK-NEXT: nop
// CHECK-NEXT: .seh_startepilogue


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
    // CHECK-LABEL: .def "?case_dtor_runs_after_join@@YAXH@Z"
    // CHECK: .seh_endprologue

    // <-- EH state -1

    // ctor call does not need a NOP, because it has real instructions after it
    HasDtor hd{42};
    // CHECK: call "??0HasDtor@@QEAA@H@Z"
    // CHECK-NEXT: test

    // <-- EH state transition from -1 0
    if (x) {
        might_throw(); // <-- NOP expected (at end of BB w/ EH_LABEL)
        // CHECK: call "?might_throw@@YAXXZ"
        // CHECK-NEXT: nop
    } else {
        other_func(10); // <-- NOP expected (at end of BB w/ EH_LABEL)
        // CHECK: call "?other_func@@YAXH@Z"
        // CHECK-NEXT: nop
    }
    does_not_throw();
    // <-- EH state transition 0 to -1
    // ~HasDtor() runs

    // CHECK: .seh_endproc

    // CHECK: "$ip2state$?case_dtor_runs_after_join@@YAXH@Z":
    // CHECK-NEXT: .long [[func_begin:.Lfunc_begin([0-9]+)@IMGREL]]
    // CHECK-NEXT: .long -1
    // CHECK-NEXT: .long [[tmp1:.Ltmp([0-9]+)]]@IMGREL
    // CHECK-NEXT: .long 0
    // CHECK-NEXT: .long [[tmp2:.Ltmp([0-9]+)]]@IMGREL
    // CHECK-NEXT: .long -1
}


// Check the behavior of NOP padding around tail calls.
// We do not expect to insert NOPs around tail calls.
// However, the first call (to other_func()) does get a NOP
// because it comes before .seh_startepilogue.
void case_tail_call_no_eh() {
    // CHECK-LABEL: .def "?case_tail_call_no_eh@@YAXXZ"
    // CHECK: .seh_endprologue

    // ordinary call
    other_func(10);
    // CHECK: call "?other_func@@YAXH@Z"
    // CHECK-NEXT: nop

    // tail call; no NOP padding after JMP
    does_not_throw();

    // CHECK: .seh_startepilogue
    // CHECK: .seh_endepilogue
    // CHECK: jmp "?does_not_throw@@YAXXZ"
    // CHECK-NOT: nop
    // CHECK: .seh_endproc
}


// Check the behavior of a try/catch
int case_try_catch() {
    // CHECK-LABEL: .def "?case_try_catch@@YAHXZ"
    // CHECK: .seh_endprologue

    // Because of the EH_LABELs, the ctor and other_func() get NOPs.

    int result = 0;
    try {
        // CHECK: call "??0HasDtor@@QEAA@H@Z"
        // CHECK-NEXT: nop
        HasDtor hd{20};

        // CHECK: call "?other_func@@YAXH@Z"
        // CHECK-NEXT: nop
        other_func(10);

        // CHECK: call "??1HasDtor@@QEAA@XZ"
        // CHECK: mov
    } catch (BadError& e) {
        result = 1;
    }
    return result;

    // CHECK: .seh_endproc

    // CHECK: .def "?dtor$4@?0??case_try_catch@@YAHXZ@4HA"
    // CHECK: .seh_endprologue
    // CHECK: call "??1HasDtor@@QEAA@XZ"
    // CHECK-NEXT: nop
    // CHECK: .seh_startepilogue
    // CHECK: .seh_endproc
}
