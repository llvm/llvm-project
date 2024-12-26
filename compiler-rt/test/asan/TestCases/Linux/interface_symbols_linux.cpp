// Check the presence of interface symbols in compiled file.
// If you're changing this file, please also change
// ../Darwin/interface_symbols_darwin.cpp

// RUN: %clangxx -x c++-header -o - -E %p/../../../../lib/asan/asan_interface.inc  \
// RUN:  | sed "s/INTERFACE_FUNCTION/\nINTERFACE_FUNCTION/g" >  %t.asan_interface.inc
// RUN: %clangxx_asan -O2 %s -o %t.exe
// RUN: nm -D %t.exe | grep " [TWw] "                                          \
// RUN:  | grep -o "\(__asan_\|__ubsan_\|__sancov_\|__sanitizer_\)[^ ]*"       \
// RUN:  | grep -v "__sanitizer_syscall"                                       \
// RUN:  | grep -v "__sanitizer_weak_hook"                                     \
// RUN:  | grep -v "__sancov_lowest_stack"                                     \
// RUN:  | sed -e "s/__asan_version_mismatch_check_v[0-9]+/__asan_version_mismatch_check/" \
// RUN:  > %t.exports
//
// RUN: sed ':a;N;$!ba;s/([\n ]*/(/g'                                              \
// RUN:  %t.asan_interface.inc                                                     \
// RUN:  %p/../../../../lib/ubsan/ubsan_interface.inc                              \
// RUN:  %p/../../../../lib/sanitizer_common/sanitizer_common_interface.inc        \
// RUN:  %p/../../../../lib/sanitizer_common/sanitizer_common_interface_posix.inc  \
// RUN:  %p/../../../../lib/sanitizer_common/sanitizer_coverage_interface.inc      \
// RUN:  | grep -e "INTERFACE_\(WEAK_\)\?FUNCTION"                                 \
// RUN:  | grep -v "__sanitizer_weak_hook"                                         \
// RUN:  | grep -v "__sanitizer_override_function"                                 \
// RUN:  | grep -v "__sanitizer_override_function_by_addr"                         \
// RUN:  | grep -v "__ubsan_handle_dynamic_type_cache_miss"                        \
// RUN:  | grep -v "__ubsan_handle_dynamic_type_cache_miss_abort"                  \
// RUN:  | grep -v "__sanitizer_register_weak_function"                            \
// RUN:  | sed -e "s/.*(//" -e "s/).*//" > %t.imports
//
// RUN: cat %t.imports | sort | uniq > %t.imports-sorted
// RUN: cat %t.exports | sort | uniq > %t.exports-sorted
//
// RUN: echo
// RUN: echo "=== NOTE === If you see a mismatch below, please update sanitizer_interface.inc files."
// RUN: diff %t.imports-sorted %t.exports-sorted
//
// FIXME: nm -D on powerpc somewhy shows ASan interface symbols residing
// in "initialized data section".
// REQUIRES: x86-target-arch,asan-static-runtime

int main() { return 0; }
