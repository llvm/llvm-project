// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature '' -target-feature m %s 2>&1 \
// RUN:    | FileCheck -check-prefix=NO_PREFIX %s

// NO_PREFIX: warning: feature flag 'm' must start with either '+' to enable the feature or '-' to disable it; flag ignored [-Winvalid-command-line-argument]

// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature '' -target-feature ' n' %s 2>&1 \
// RUN:    | FileCheck -check-prefix=NULL_PREFIX %s

// NULL_PREFIX: warning: feature flag ' n' must start with either '+' to enable the feature or '-' to disable it; flag ignored [-Winvalid-command-line-argument]
