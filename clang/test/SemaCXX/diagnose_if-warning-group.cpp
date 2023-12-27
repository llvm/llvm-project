// RUN: %clang_cc1 %s -verify -fno-builtin -Werror=comment -Wno-error=abi

#define _diagnose_if(...) __attribute__((diagnose_if(__VA_ARGS__)))

template <bool b>
void diagnose_if_wcomma() _diagnose_if(b, "oh no", "warning", "comma") {}

template <bool b>
void diagnose_if_wcomment() _diagnose_if(b, "oh no", "warning", "comment") {}

void bougus_warning() _diagnose_if(true, "oh no", "warning", "bougus warning") {} // expected-error {{unknown warning group}}

void show_in_system_header() _diagnose_if(true, "oh no", "warning", "assume", "Banane") {} // expected-error {{'diagnose_if' attribute takes no more than 4 arguments}}


void diagnose_if_wabi_default_error() _diagnose_if(true, "ABI stuff", "error", "abi") {}

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
}
