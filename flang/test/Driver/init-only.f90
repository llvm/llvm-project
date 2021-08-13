! Verify that -init-only flag generates a diagnostic as expected

! RUN: %flang_fc1 -init-only 2>&1 | FileCheck %s

! CHECK: warning: Use `-init-only` for testing purposes only
