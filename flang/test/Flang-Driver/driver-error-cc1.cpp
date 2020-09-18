// RUN: not %flang-new %s 2>&1 | FileCheck %s

// REQUIRES: new-flang-driver

// C++ files are currently not supported (i.e. `flang -cc1`)

// CHECK:error: unknown integrated tool '-cc1'. Valid tools include '-fc1'.
