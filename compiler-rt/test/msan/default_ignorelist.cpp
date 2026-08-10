// Test that MSan uses the default ignorelist from resource directory.
// RUN: %clangxx_msan -### %s 2>&1 | FileCheck %s
// CHECK: fsanitize-system-ignorelist={{.*}}msan_ignorelist.txt
