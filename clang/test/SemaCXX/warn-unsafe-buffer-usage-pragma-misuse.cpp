// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions -verify %s

void beginUnclosed(int * x) {
#pragma clang unsafe_buffer_usage begin

#pragma clang unsafe_buffer_usage begin  // expected-error{{already inside '#pragma unsafe_buffer_usage'}}
  x++;
#pragma clang unsafe_buffer_usage end
}

void endUnopened(int *x) {
#pragma clang unsafe_buffer_usage end    // expected-error{{not currently inside '#pragma unsafe_buffer_usage'}}

#pragma clang unsafe_buffer_usage begin
  x++;
#pragma clang unsafe_buffer_usage end
}

void wrongOption() {
#pragma clang unsafe_buffer_usage start // expected-error{{Expected 'begin' or 'end'}}
#pragma clang unsafe_buffer_usage close // expected-error{{Expected 'begin' or 'end'}}
}

void unclosed(int * p1) {
#pragma clang unsafe_buffer_usage begin
// End of the included file will not raise the unclosed region warning:
#define _INCLUDE_NO_WARN
#include "warn-unsafe-buffer-usage-pragma.h"
#pragma clang unsafe_buffer_usage end

// End of this file raises the warning:
#pragma clang unsafe_buffer_usage begin  // expected-error{{'#pragma unsafe_buffer_usage' was not ended}}
}
