// RUN: %clang_cc1 %s -verify=expected,wall -fno-builtin -Wno-pedantic -Werror=comment -Wno-error=abi -Wfatal-errors=assume -Wno-fatal-errors=assume -Wno-format
// RUN: %clang_cc1 %s -verify=expected,wno-all,pedantic,format -fno-builtin -Wno-all -Werror=comment -Wno-error=abi -Werror=assume -Wformat

#define diagnose_if(...) __attribute__((diagnose_if(__VA_ARGS__)))

#ifndef EMTY_WARNING_GROUP
void bougus_warning() diagnose_if(true, "oh no", "warning", "bogus warning") {} // expected-error {{unknown warning group 'bogus warning'}}

void show_in_system_header() diagnose_if(true, "oh no", "warning", "assume", "Banane") {} // expected-error {{'diagnose_if' attribute takes no more than 4 arguments}}
#endif // EMTY_WARNING_GROUP

template <bool b>
void diagnose_if_wcomma() diagnose_if(b, "oh no", "warning", "comma") {}

template <bool b>
void diagnose_if_wcomment() diagnose_if(b, "oh no", "warning", "comment") {}

void empty_warning_group() diagnose_if(true, "oh no", "warning", "") {} // expected-error {{unknown warning group ''}}
void empty_warning_group_error() diagnose_if(true, "oh no", "error", "") {} // expected-error {{unknown warning group ''}}

void diagnose_if_wabi_default_error() diagnose_if(true, "ABI stuff", "error", "abi") {}
void diagnose_assume() diagnose_if(true, "Assume diagnostic", "warning", "assume") {}

void Wall() diagnose_if(true, "oh no", "warning", "all") {}
void Wpedantic() diagnose_if(true, "oh no", "warning", "pedantic") {}
void Wformat_extra_args() diagnose_if(true, "oh no", "warning", "format-extra-args") {}

void call() {
  diagnose_if_wcomma<true>(); // expected-warning {{oh no}}
  diagnose_if_wcomma<false>();
  diagnose_if_wcomment<true>(); // expected-error {{oh no}}
  diagnose_if_wcomment<false>();

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcomma"
  diagnose_if_wcomma<true>();
  diagnose_if_wcomment<true>(); // expected-error {{oh no}}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcomment"
  diagnose_if_wcomma<true>(); // expected-warning {{oh no}}
  diagnose_if_wcomment<true>();
#pragma clang diagnostic pop

  diagnose_if_wcomma<true>(); // expected-warning {{oh no}}
  diagnose_if_wcomment<true>(); // expected-error {{oh no}}

  diagnose_if_wabi_default_error(); // expected-warning {{ABI stuff}}
  diagnose_assume(); // expected-error {{Assume diagnostic}}

  // Make sure that the -Wassume diagnostic isn't fatal
  diagnose_if_wabi_default_error(); // expected-warning {{ABI stuff}}

  Wall(); // wall-warning {{oh no}}
  Wpedantic(); // pedantic-warning {{oh no}}
  Wformat_extra_args(); // format-warning {{oh no}}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat"
  Wformat_extra_args();
#pragma clang diagnostic pop
}
