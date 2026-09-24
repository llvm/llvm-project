// Test the printing and parsing of the fir.allocation_policy module attribute.
// All parameters are mandatory so that the concrete policy is preserved across
// an IR round trip.

// RUN: fir-opt %s | fir-opt | FileCheck %s

// CHECK-LABEL: module @all_fields
// CHECK-SAME:  fir.allocation_policy = #fir.allocation_policy<stack_arrays = true,
// CHECK-SAME:  small_array_threshold = 64, total_stack_limit = 128>
module @all_fields attributes {fir.allocation_policy =
    #fir.allocation_policy<stack_arrays = true, small_array_threshold = 64,
                           total_stack_limit = 128>} {
}

// Check a policy containing the values used by the construction helpers when
// no options override them.
// CHECK-LABEL: module @default_values
// CHECK-SAME:  fir.allocation_policy = #fir.allocation_policy<stack_arrays = false,
// CHECK-SAME:  small_array_threshold = 1024, total_stack_limit = 4194304>
module @default_values attributes {fir.allocation_policy =
    #fir.allocation_policy<stack_arrays = false,
                           small_array_threshold = 1024,
                           total_stack_limit = 4194304>} {
}
