// Check the presence of interface symbols in compiled file.

// RUN: %clang_asan -O2 %s -o %t.exe
// RUN: nm -D %t.exe | grep " T " | sed "s/.* T //" \
// RUN:    | grep "__asan_" | sed "s/___asan_/__asan_/" \
// RUN:    | sed -E "s/__asan_version_mismatch_check_v[0-9]+/__asan_version_mismatch_check/" \
// RUN:    | grep -v "__asan_default_options" \
// RUN:    | grep -v "__asan_stack_" \
// RUN:    | grep -v "__asan_on_error" > %t.symbols
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
// RUN: echo __asan_alloca_poison >> %t.interface
// RUN: echo __asan_allocas_unpoison >> %t.interface
// RUN: cat %t.interface | sort -u | diff %t.symbols -

// FIXME: nm -D on powerpc somewhy shows ASan interface symbols residing
// in "initialized data section".
// REQUIRES: x86-target-arch,asan-static-runtime

int main() { return 0; }
