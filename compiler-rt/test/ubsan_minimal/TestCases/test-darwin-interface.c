// Check that the ubsan and ubsan-minimal runtimes have the same symbols,
// making exceptions as necessary.
//
// REQUIRES: x86_64-darwin

// RUN: nm -jgU `%clangxx -fsanitize-minimal-runtime -fsanitize=undefined %s -o %t '-###' 2>&1 | grep "libclang_rt.ubsan_minimal_osx_dynamic.dylib" | sed -e 's/.*"\(.*libclang_rt.ubsan_minimal_osx_dynamic.dylib\)".*/\1/'` | grep "^___ubsan_handle" \
// RUN:  | sed 's/_minimal//g' \
// RUN:  > %t.minimal.symlist
//
// RUN: nm -jgU `%clangxx -fno-sanitize-minimal-runtime -fsanitize=undefined %s -o %t '-###' 2>&1 | grep "libclang_rt.ubsan_osx_dynamic.dylib" | sed -e 's/.*"\(.*libclang_rt.ubsan_osx_dynamic.dylib\)".*/\1/'` | grep "^___ubsan_handle" \
// RUN:  | grep -vE "^___ubsan_handle_dynamic_type_cache_miss" \
// RUN:  | grep -vE "^___ubsan_handle_cfi_bad_type" \
// RUN:  | sed 's/_v1//g' \
// RUN:  > %t.full.symlist
//
// RUN: diff %t.minimal.symlist %t.full.symlist
