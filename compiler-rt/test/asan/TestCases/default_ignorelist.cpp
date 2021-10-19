// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// XFAIL: android
// UNSUPPORTED: ios
//
// Test that ASan uses the default ignorelist from resource directory.
// RUN: %clangxx_asan -### %s 2>&1 | FileCheck %s
// CHECK: fsanitize-system-ignorelist={{.*}}asan_ignorelist.txt
