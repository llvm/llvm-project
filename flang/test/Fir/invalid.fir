// RUN: fir-opt -split-input-file -verify-diagnostics %s

// expected-error@+2{{expected integer value}}
// expected-error@+1{{kind value expected}}
func private @it1() -> !fir.int<A>

// -----

// expected-error@+1{{custom op 'fir.string_lit' must have character type}}
%0 = fir.string_lit "Hello, World!"(13) : !fir.int<32>

// -----

// expected-error@+1{{custom op 'fir.string_lit' found an invalid constant}}
%0 = fir.string_lit 20(13) : !fir.int<32>

// -----

// expected-error@+1{{'fir.string_lit' op values in list must be integers}}
%2 = fir.string_lit [158, 2.0](2) : !fir.char<2>