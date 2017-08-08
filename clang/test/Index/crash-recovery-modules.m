// Clear out the module cache entirely, so we start from nothing.
// RUN: rm -rf %t

// Parse the file, such that building the module will cause Clang to crash.
// RUN: not env CINDEXTEST_FAILONERROR=1 c-index-test -test-load-source all -fmodules -fmodules-cache-path=%t -Xclang -fdisable-module-hash -I %S/Inputs/Headers -DCRASH %s > /dev/null 2> %t.err
// RUN: FileCheck < %t.err -check-prefix=CHECK-CRASH %s
// CHECK-CRASH: crash-recovery-modules.m:17:9:{17:2-17:14}: fatal error: could not build module 'Crash'

// Parse the file again, without crashing, to make sure that
// subsequent parses do the right thing.
// RUN: env CINDEXTEST_FAILONERROR=1 c-index-test -test-load-source all -fmodules -fmodules-cache-path=%t -Xclang -fdisable-module-hash -I %S/Inputs/Headers %s > /dev/null

// REQUIRES: crash-recovery
// REQUIRES: shell
// UNSUPPORTED: libstdcxx-safe-mode

@import Crash;

#ifdef LIBCLANG_CRASH
#pragma clang __debug crash
#endif

void test() {
  const char* error = getCrashString();
}


// RUN: rm -rf %t
// Check that libclang crash-recovery works; both with a module building crash...
// RUN: not env CINDEXTEST_FAILONERROR=1 c-index-test -test-load-source all -fmodules -fmodules-cache-path=%t -Xclang -fdisable-module-hash -I %S/Inputs/Headers -DCRASH -DLIBCLANG_CRASH %s > /dev/null 2> %t.err
// RUN: FileCheck < %t.err -check-prefix=CHECK-LIBCLANG-CRASH %s
// ...and with module building successful.
// RUN: not env CINDEXTEST_FAILONERROR=1 c-index-test -test-load-source all -fmodules -fmodules-cache-path=%t -Xclang -fdisable-module-hash -I %S/Inputs/Headers -DLIBCLANG_CRASH %s > /dev/null 2> %t.err
// RUN: FileCheck < %t.err -check-prefix=CHECK-LIBCLANG-CRASH %s
// CHECK-LIBCLANG-CRASH: libclang: crash detected during parsing
// CHECK-LIBCLANG-CRASH: Unable to load translation unit!
