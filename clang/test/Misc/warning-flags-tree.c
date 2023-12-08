// RUN: diagtool tree --internal | FileCheck -strict-whitespace %s
// RUN: diagtool tree --internal -Weverything | FileCheck -strict-whitespace %s
// RUN: diagtool tree --internal everything | FileCheck -strict-whitespace %s
//
// These three ways of running diagtool tree are the same:
// they produce a tree for every top-level diagnostic flag.
// Just check a few to make sure we're actually showing more than one group.
//
// CHECK: -W
// CHECK:   -Wextra
// CHECK:     -Wmissing-field-initializers
// CHECK:       warn_missing_field_initializers
// CHECK: -Wall
// CHECK:   -Wmost

// These flags are currently unimplemented; test that we output them anyway.
// CHECK: -Wstrict-aliasing
// CHECK-NEXT: -Wstrict-aliasing=0
// CHECK-NEXT: -Wstrict-aliasing=1
// CHECK-NEXT: -Wstrict-aliasing=2
// CHECK: -Wstrict-overflow
// CHECK-NEXT: -Wstrict-overflow=0
// CHECK-NEXT: -Wstrict-overflow=1
// CHECK-NEXT: -Wstrict-overflow=2
// CHECK-NEXT: -Wstrict-overflow=3
// CHECK-NEXT: -Wstrict-overflow=4
// CHECK-NEXT: -Wstrict-overflow=5


// RUN: not diagtool tree -Wthis-is-not-a-valid-flag

// RUN: diagtool tree --internal -Wgnu | FileCheck -strict-whitespace -check-prefix CHECK-GNU %s
// CHECK-GNU: -Wgnu
// CHECK-GNU:   -Wgnu-designator
// CHECK-GNU:     ext_gnu_array_range
// CHECK-GNU:     ext_gnu_missing_equal_designator
// CHECK-GNU:     ext_gnu_old_style_field_designator
// CHECK-GNU:   -Wvla-extension
// CHECK-GNU:     ext_vla
// There are more GNU extensions but we don't need to check them all.

// RUN: diagtool tree -Wgnu | FileCheck -check-prefix CHECK-FLAGS-ONLY %s
// CHECK-FLAGS-ONLY: -Wgnu
// CHECK-FLAGS-ONLY:   -Wgnu-designator
// CHECK-FLAGS-ONLY-NOT:     ext_gnu_array_range
// CHECK-FLAGS-ONLY-NOT:     ext_gnu_missing_equal_designator
// CHECK-FLAGS-ONLY-NOT:     ext_gnu_old_style_field_designator
// CHECK-FLAGS-ONLY:   -Wvla
// CHECK-FLAGS-ONLY-NOT:     ext_vla
// CHECK-FLAGS-ONLY-NOT:   ext_array_init_copy
// CHECK-FLAGS-ONLY-NOT:   ext_empty_struct_union
// CHECK-FLAGS-ONLY-NOT:   ext_expr_not_ice
