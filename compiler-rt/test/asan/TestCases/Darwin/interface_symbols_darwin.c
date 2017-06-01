// Check the presence of interface symbols in the ASan runtime dylib.
// If you're changing this file, please also change
// ../Linux/interface_symbols.c

// RUN: %clang_asan -dead_strip -O2 %s -o %t.exe
// RUN: rm -f %t.symbols %t.interface

// RUN: nm -g `%clang_asan %s -fsanitize=address -### 2>&1 | grep "libclang_rt.asan_osx_dynamic.dylib" | sed -e 's/.*"\(.*libclang_rt.asan_osx_dynamic.dylib\)".*/\1/'` \
// RUN:   | grep " T " | sed "s/.* T //" \
// RUN:   | grep "__asan_" | sed "s/___asan_/__asan_/" \
// RUN:   | sed -E "s/__asan_version_mismatch_check_v[0-9]+/__asan_version_mismatch_check/" \
// RUN:   | grep -v "__asan_default_options" \
// RUN:   | grep -v "__asan_on_error" > %t.symbols

// RUN: cat %p/../../../../lib/asan/asan_interface_internal.h \
// RUN:    | sed "s/\/\/.*//" | sed "s/typedef.*//" \
// RUN:    | grep -v "OPTIONAL" \
// RUN:    | grep "__asan_.*(" | sed "s/.* __asan_/__asan_/;s/(.*//" \
// RUN:    > %t.interface
// RUN: echo __asan_report_load1 >> %t.interface
// RUN: echo __asan_report_load2 >> %t.interface
// RUN: echo __asan_report_load4 >> %t.interface
// RUN: echo __asan_report_load8 >> %t.interface
// RUN: echo __asan_report_load16 >> %t.interface
// RUN: echo __asan_report_store1 >> %t.interface
// RUN: echo __asan_report_store2 >> %t.interface
// RUN: echo __asan_report_store4 >> %t.interface
// RUN: echo __asan_report_store8 >> %t.interface
// RUN: echo __asan_report_store16 >> %t.interface
// RUN: echo __asan_report_load_n >> %t.interface
// RUN: echo __asan_report_store_n >> %t.interface
// RUN: echo __asan_report_load1_noabort >> %t.interface
// RUN: echo __asan_report_load2_noabort >> %t.interface
// RUN: echo __asan_report_load4_noabort >> %t.interface
// RUN: echo __asan_report_load8_noabort >> %t.interface
// RUN: echo __asan_report_load16_noabort >> %t.interface
// RUN: echo __asan_report_store1_noabort >> %t.interface
// RUN: echo __asan_report_store2_noabort >> %t.interface
// RUN: echo __asan_report_store4_noabort >> %t.interface
// RUN: echo __asan_report_store8_noabort >> %t.interface
// RUN: echo __asan_report_store16_noabort >> %t.interface
// RUN: echo __asan_report_load_n_noabort >> %t.interface
// RUN: echo __asan_report_store_n_noabort >> %t.interface
// RUN: echo __asan_report_exp_load1 >> %t.interface
// RUN: echo __asan_report_exp_load2 >> %t.interface
// RUN: echo __asan_report_exp_load4 >> %t.interface
// RUN: echo __asan_report_exp_load8 >> %t.interface
// RUN: echo __asan_report_exp_load16 >> %t.interface
// RUN: echo __asan_report_exp_store1 >> %t.interface
// RUN: echo __asan_report_exp_store2 >> %t.interface
// RUN: echo __asan_report_exp_store4 >> %t.interface
// RUN: echo __asan_report_exp_store8 >> %t.interface
// RUN: echo __asan_report_exp_store16 >> %t.interface
// RUN: echo __asan_report_exp_load_n >> %t.interface
// RUN: echo __asan_report_exp_store_n >> %t.interface
// RUN: echo __asan_get_current_fake_stack >> %t.interface
// RUN: echo __asan_addr_is_in_fake_stack >> %t.interface

// RUN: for i in `jot - 0 10`; do echo __asan_stack_malloc_$i >> %t.interface; done
// RUN: for i in `jot - 0 10`; do echo __asan_stack_free_$i >> %t.interface; done

// RUN: cat %t.interface | sort -u | diff %t.symbols -

int main() { return 0; }
