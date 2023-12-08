// REQUIRES: crash-recovery
// RUN: rm -rf %t

// Crash building module.
// RUN: not --crash %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs %s

// The dead symlink is still around, but the underlying lock file is gone.
// RUN: find %t -name "crash-*.pcm.lock" | count 1
// RUN: find %t -name "crash-*.pcm.lock-*" | count 0

@import crash;
