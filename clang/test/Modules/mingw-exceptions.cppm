// REQUIRES: x86-registered-target
// RUN: %clang -target x86_64-windows-gnu -x c++-module -std=gnu++23 -fno-modules-reduced-bmi \
// RUN:     -c -o /dev/null -Xclang -disable-llvm-passes %s

// Make sure the command succeeds and doesn't break on the -exception-model flag in cc1.
export module empty;
