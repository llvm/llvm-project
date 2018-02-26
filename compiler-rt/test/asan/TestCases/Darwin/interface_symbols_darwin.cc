// Check the presence of interface symbols in the ASan runtime dylib.
// If you're changing this file, please also change
// ../Linux/interface_symbols.c

// RUN: %clangxx_asan -dead_strip -O2 %s -o %t.exe
//
// note: we can not use -D on Darwin.
// RUN: nm -g `%clang_asan %s -fsanitize=address -### 2>&1 | grep "libclang_rt.asan_osx_dynamic.dylib" | sed -e 's/.*"\(.*libclang_rt.asan_osx_dynamic.dylib\)".*/\1/'` \
// RUN:  | grep " [TU] "                                                       \
// RUN:  | grep -o "\(__asan_\|__ubsan_\|__sancov_\|__sanitizer_\)[^ ]*"       \
// RUN:  | grep -v "__sanitizer_syscall"                                       \
// RUN:  | grep -v "__sanitizer_weak_hook"                                     \
// RUN:  | grep -v "__sanitizer_mz"                                            \
// RUN:  | grep -v "__sancov_lowest_stack"                                     \
// RUN:  | sed -e "s/__asan_version_mismatch_check_v[0-9]+/__asan_version_mismatch_check/" \
// RUN:  > %t.exports
//
// RUN: grep -e "INTERFACE_\(WEAK_\)\?FUNCTION"                                \
// RUN:  %p/../../../../lib/asan/asan_interface.inc                            \
// RUN:  %p/../../../../lib/ubsan/ubsan_interface.inc                          \
// RUN:  %p/../../../../lib/sanitizer_common/sanitizer_common_interface.inc    \
// RUN:  %p/../../../../lib/sanitizer_common/sanitizer_common_interface_posix.inc \
// RUN:  %p/../../../../lib/sanitizer_common/sanitizer_coverage_interface.inc  \
// RUN:  | grep -v "__sanitizer_weak_hook"                                     \
// RUN:  | sed -e "s/.*(//" -e "s/).*//" > %t.imports
//
// RUN: cat %t.imports | sort | uniq > %t.imports-sorted
// RUN: cat %t.exports | sort | uniq > %t.exports-sorted
//
// RUN: echo
// RUN: echo "=== NOTE === If you see a mismatch below, please update sanitizer_interface.inc files."
// RUN: diff %t.imports-sorted %t.exports-sorted

// UNSUPPORTED: ios

int main() { return 0; }
