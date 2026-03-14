// -----------------------------------------------------------------------------
// Tests for the -mvscale-min flag
// -----------------------------------------------------------------------------

// Error out if value is unbounded.
// -----------------------------------------------------------------------------
// RUN: not %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN:  -mvscale-min=0 2>&1 | FileCheck %s

// CHECK: error: minimum vscale must be an unsigned integer greater than 0
