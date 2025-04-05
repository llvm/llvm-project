// Check that -Xclangas passes args to -cc1as.
// RUN: %clang -### -Xclangas -target-feature -Xclangas +v5t %s 2>&1 | \
// RUN:   FileCheck -check-prefix=ARGS %s
// ARGS: -cc1as
// ARGS: -target-feature
// ARGS: +v5t
