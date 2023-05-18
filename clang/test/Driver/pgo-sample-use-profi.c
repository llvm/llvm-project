/// Test if profi flat is enabled in frontend as user-facing feature.
// RUN: %clang --target=x86_64 -c -fsample-profile-use-profi -fprofile-sample-use=/dev/null -### %s 2>&1 | FileCheck %s

// CHECK: "-mllvm" "-sample-profile-use-profi"
