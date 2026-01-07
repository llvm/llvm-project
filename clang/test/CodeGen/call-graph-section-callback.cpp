// Tests that callback function whose address is taken is attached Type ID metadata
// as it is a potential indirect call target.

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -emit-llvm -o %t %s
// RUN: FileCheck %s < %t

////////////////////////////////////////////////////////////////////////////////
typedef void (*CallbackFn)(int);

// Callback function with "internal" linkage.
// CHECK-LABEL: define internal void @_ZL10myCallbacki(
// CHECK-SAME: {{.*}} !type [[F_CALLBACK:![0-9]+]]
static void myCallback(int value) 
{
    volatile int sink = value;
    (void)sink;
}

int takeCallbackAddress() {
    // Take the address of the callback explicitly (address-taken function)
    CallbackFn cb = &myCallback;
    // Store the address in a volatile pointer to keep it observable
    volatile void* addr = (void*)cb;
    (void)addr;

    return 0;
}

// CHECK: [[F_CALLBACK]]   = !{i64 0, !"_ZTSFviE.generalized"}
