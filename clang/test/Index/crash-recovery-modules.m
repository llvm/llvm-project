// Clear out the module cache entirely, so we start from nothing.
// RUN: rm -rf %t

// Parse the file, such that building the module will cause Clang to crash.
// RUN: env CINDEXTEST_FAILONERROR=1 not c-index-test -test-load-source all -fmodules -fmodules-cache-path=%t -Xclang -fdisable-module-hash -I %S/Inputs/Headers -DCRASH %s > /dev/null 2> %t.err
// RUN: FileCheck < %t.err -check-prefix=CHECK-CRASH %s
// CHECK-CRASH: crash-recovery-modules.m:16:9:{16:2-16:14}: fatal error: could not build module 'Crash'

// Parse the file again, without crashing, to make sure that
// subsequent parses do the right thing.
// RUN: env CINDEXTEST_FAILONERROR=1 c-index-test -test-load-source all -fmodules -fmodules-cache-path=%t -Xclang -fdisable-module-hash -I %S/Inputs/Headers %s > /dev/null

// REQUIRES: crash-recovery
// UNSUPPORTED: libstdcxx-safe-mode, hwasan

@import Crash;

#ifdef LIBCLANG_CRASH
#pragma clang __debug crash
#endif

void test(void) {
  const char* error = getCrashString();
}


// RUN: rm -rf %t
// Check that libclang crash-recovery works; both with a module building crash...
// RUN: env CINDEXTEST_FAILONERROR=1 not c-index-test -test-load-source all -fmodules -fmodules-cache-path=%t -Xclang -fdisable-module-hash -I %S/Inputs/Headers -DCRASH -DLIBCLANG_CRASH %s > /dev/null 2> %t.err
// RUN: FileCheck < %t.err -check-prefix=CHECK-LIBCLANG-CRASH %s
// ...and with module building successful.
// RUN: env CINDEXTEST_FAILONERROR=1 not c-index-test -test-load-source all -fmodules -fmodules-cache-path=%t -Xclang -fdisable-module-hash -I %S/Inputs/Headers -DLIBCLANG_CRASH %s > /dev/null 2> %t.err
// RUN: FileCheck < %t.err -check-prefix=CHECK-LIBCLANG-CRASH %s
// CHECK-LIBCLANG-CRASH-DAG: libclang: crash detected during parsing
// CHECK-LIBCLANG-CRASH-DAG: Unable to load translation unit!
