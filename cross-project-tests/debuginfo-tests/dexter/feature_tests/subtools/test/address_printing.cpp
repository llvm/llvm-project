// Purpose:
//      Test that address values in a \DexExpectWatchValue are printed with
//      their address name along with the address' resolved value (if any), and
//      that when verbose output is enabled the complete map of resolved
//      addresses and list of unresolved addresses will also be printed.
//
//      Note: Currently "misordered result" is the only penalty that does not
//      display the address properly; if it is implemented, this test should be
//      updated.
//
// The dbgeng driver doesn't support \DexLimitSteps yet.
// UNSUPPORTED: system-windows
//
// RUN: %dexter_regression_test_build %s -o %t
// RUN: not %dexter_regression_test_run --binary %t -v -- %s | FileCheck %s

// CHECK: Resolved Addresses:
// CHECK-NEXT: 'x_2': 0x[[X2_VAL:[0-9a-f]+]]
// CHECK-NEXT: 'y': 0x[[Y_VAL:[0-9a-f]+]]
// CHECK: Unresolved Addresses:
// CHECK-NEXT: ['x_1']

// CHECK-LABEL: [x] ExpectValue
// CHECK: expected encountered watches:
// CHECK-NEXT: address 'x_2' (0x[[X2_VAL]])
// CHECK: missing values:
// CHECK-NEXT: address 'x_1'

// CHECK-LABEL: [z] ExpectValue
// CHECK: expected encountered watches:
// CHECK-NEXT: address 'x_2' (0x[[X2_VAL]])
// CHECK-NEXT: address 'y' (0x[[Y_VAL]])
// CHECK: misordered result:
// CHECK-NEXT: (0x[[Y_VAL]]): step 4
// CHECK-NEXT: (0x[[X2_VAL]]): step 5

int main() {
    int *x = new int(5);
    int *y = new int(4);
    if (false) {
        (void)0; // DexLabel('unreachable')
    }
    int *z = y;
    z = x; // DexLabel('start_line')
    delete y;
    delete x; // DexLabel('end_line')
}

// DexDeclareAddress('x_1', 'x', on_line=ref('unreachable'))
// DexDeclareAddress('x_2', 'x', on_line=ref('end_line'))
// DexDeclareAddress('y', 'y', on_line=ref('start_line'))
// DexExpectWatchValue('x', address('x_1'), address('x_2'), from_line=ref('start_line'), to_line=ref('end_line'))
// DexExpectWatchValue('z', address('x_2'), address('y'), from_line=ref('start_line'), to_line=ref('end_line'))
