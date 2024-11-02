// Check that the interface exported by asan static lib matches the list of
// functions mentioned in sanitizer_interface.inc.
//
// Just make sure we can compile this.
// RUN: %clang_cl_asan -Od %s -Fe%t
//
// note: The mangling decoration (i.e. @4 )is removed because calling convention
//       differ from 32-bit and 64-bit.
//
// RUN: dumpbin /EXPORTS %t | sed "s/=.*//"                                    \
// RUN:   | grep -o "\(__asan_\|__ubsan_\|__sanitizer_\|__sancov_\)[^ ]*"      \
// RUN:   | grep -v "__asan_wrap"                                              \
// RUN:   | sed -e s/@.*// > %t.exports
//
// [BEWARE: be really careful with the sed commands, as this test can be run
//  from different environments with different shells and seds]
//
// RUN: sed ':a;N;$!ba;s/([\n ]*/(/g'                                          \
// RUN:  %p/../../../../lib/asan/asan_interface.inc                            \
// RUN:  %p/../../../../lib/ubsan/ubsan_interface.inc                          \
// RUN:  %p/../../../../lib/sanitizer_common/sanitizer_common_interface.inc    \
// RUN:  %p/../../../../lib/sanitizer_common/sanitizer_coverage_interface.inc  \
// RUN:  | grep -e "^INTERFACE_FUNCTION"                                       \
// RUN:  | sed -e "s/.*(//" -e "s/).*//" > %t.imports1
//
// RUN: sed ':a;N;$!ba;s/([\n ]*/(/g'                                          \
// RUN:  %p/../../../../lib/asan/asan_interface.inc                            \
// RUN:  %p/../../../../lib/ubsan/ubsan_interface.inc                          \
// RUN:  %p/../../../../lib/sanitizer_common/sanitizer_common_interface.inc    \
// RUN:  %p/../../../../lib/sanitizer_common/sanitizer_coverage_interface.inc  \
// RUN:  | grep -e "^INTERFACE_WEAK_FUNCTION"                                  \
// RUN:  | sed -e "s/.*(//" -e "s/).*/__dll/" > %t.imports2
//
// Add functions not included in the interface lists:
// RUN: grep '[I]MPORT:' %s | sed -e 's/.*[I]MPORT: //' > %t.imports3
// IMPORT: __asan_shadow_memory_dynamic_address
// IMPORT: __asan_get_shadow_memory_dynamic_address
// IMPORT: __asan_option_detect_stack_use_after_return
// IMPORT: __asan_should_detect_stack_use_after_return
// IMPORT: __asan_set_seh_filter
// IMPORT: __asan_unhandled_exception_filter
// IMPORT: __asan_test_only_reported_buggy_pointer
// IMPORT: __sancov_lowest_stack
// IMPORT: __ubsan_vptr_type_cache
//
// RUN: cat %t.imports1 %t.imports2 %t.imports3 | sort | uniq > %t.imports-sorted
// RUN: cat %t.exports | sort | uniq > %t.exports-sorted
//
// Now make sure the DLL thunk imports everything:
// RUN: echo
// RUN: echo "=== NOTE === If you see a mismatch below, please update interface.inc files."
// RUN: diff %t.imports-sorted %t.exports-sorted
// REQUIRES: asan-static-runtime

int main() { return 0; }
