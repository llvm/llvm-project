#pragma clang assume_nonnull start // expected-error{{expected 'begin' or 'end'}}

#pragma clang assume_nonnull begin // expected-note{{#pragma entered here}}

#include "nullability-pragmas-3.h" // expected-error{{cannot #include files inside '#pragma clang assume_nonnull'}}

#pragma clang assume_nonnull begin // expected-note{{#pragma entered here}}
#pragma clang assume_nonnull begin // expected-error{{already inside '#pragma clang assume_nonnull'}}
#pragma clang assume_nonnull end

#pragma clang assume_nonnull begin // expected-error{{'#pragma clang assume_nonnull' was not ended within this file}}

