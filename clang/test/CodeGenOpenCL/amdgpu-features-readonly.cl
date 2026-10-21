// RUN: %clang_cc1 -triple amdgpu9.42 -target-feature +gws -o /dev/null %s 2>&1 \
// RUN:   | FileCheck --check-prefix=GWS %s

// GWS: warning: feature flag '+gws' is ignored since the feature is read only [-Winvalid-command-line-argument]

// RUN: %clang_cc1 -triple amdgpu9.06 -target-feature +sramecc-on-off-modes -o /dev/null %s 2>&1 \
// RUN:   | FileCheck --check-prefix=SRAMECC-MODES %s
// RUN: %clang_cc1 -triple amdgpu9.06 -target-feature -sramecc-on-off-modes -o /dev/null %s 2>&1 \
// RUN:   | FileCheck --check-prefix=SRAMECC-MODES %s

// SRAMECC-MODES: warning: feature flag '{{[+-]}}sramecc-on-off-modes' is ignored since the feature is read only [-Winvalid-command-line-argument]

kernel void test() {}
