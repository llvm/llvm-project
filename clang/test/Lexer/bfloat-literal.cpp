// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -triple x86_64 -DSUPPORTED %s
// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -triple armv7 %s
// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -triple armv7 -target-feature +bf16 -DSUPPORTED %s

#ifdef SUPPORTED

__bf16 a = 1.b; // expected-error{{invalid suffix 'b' on floating constant}}
__bf16 b = 1.bf; // expected-error{{invalid suffix 'bf' on floating constant}}
__bf16 c = 1.bf166; // expected-error{{invalid suffix 'bf166' on floating constant}}
__bf16 d = 1.bf1; // expected-error{{invalid suffix 'bf1' on floating constant}}

__bf16 e = 1.B; // expected-error{{invalid suffix 'B' on floating constant}}
__bf16 f = 1.BF; // expected-error{{invalid suffix 'BF' on floating constant}}
__bf16 g = 1.BF166; // expected-error{{invalid suffix 'BF166' on floating constant}}
__bf16 h = 1.BF1; // expected-error{{invalid suffix 'BF1' on floating constant}}

__bf16 i = 1.bf16; // expect-success
__bf16 j = 1.BF16; // expect-success

__bf16 k = 1B; // expected-error{{invalid digit 'B' in decimal constant}}
__bf16 l = 1BF; // expected-error{{invalid digit 'B' in decimal constant}}
__bf16 m = 1BF166; // expected-error{{invalid digit 'B' in decimal constant}}
__bf16 n = 1BF1; // expected-error{{invalid digit 'B' in decimal constant}}

__bf16 o = 1b; // expected-error{{invalid digit 'b' in decimal constant}}
__bf16 p = 1bf; // expected-error{{invalid digit 'b' in decimal constant}}
__bf16 q = 1bf166; // expected-error{{invalid digit 'b' in decimal constant}}
__bf16 r = 1bf1; // expected-error{{invalid digit 'b' in decimal constant}}

__bf16 s = 1bf16; // expected-error{{invalid digit 'b' in decimal constant}}
__bf16 t = 1BF16; // expected-error{{invalid digit 'B' in decimal constant}}

__bf16 u = 1.bf16F16; // expected-error{{invalid suffix 'bf16F16' on floating constant}}
__bf16 v = 1.BF16f16; // expected-error{{invalid suffix 'BF16f16' on floating constant}}
__bf16 w = 1.F16bf16; // expected-error{{invalid suffix 'F16bf16' on floating constant}}

#endif

#ifndef SUPPORTED

__bf16 a = 1.b; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid suffix 'b' on floating constant}}
__bf16 b = 1.bf; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid suffix 'bf' on floating constant}}
__bf16 c = 1.bf166; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid suffix 'bf166' on floating constant}}
__bf16 d = 1.bf1; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid suffix 'bf1' on floating constant}}

__bf16 e = 1.B; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid suffix 'B' on floating constant}}
__bf16 f = 1.BF; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid suffix 'BF' on floating constant}}
__bf16 g = 1.BF166; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid suffix 'BF166' on floating constant}}
__bf16 h = 1.BF1; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid suffix 'BF1' on floating constant}}

__bf16 i = 1.bf16; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid suffix 'bf16' on floating constant}}
__bf16 j = 1.BF16; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid suffix 'BF16' on floating constant}}

__bf16 k = 1B; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid digit 'B' in decimal constant}}
__bf16 l = 1BF; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid digit 'B' in decimal constant}}
__bf16 m = 1BF166; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid digit 'B' in decimal constant}}
__bf16 n = 1BF1; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid digit 'B' in decimal constant}}

__bf16 o = 1b; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid digit 'b' in decimal constant}}
__bf16 p = 1bf; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid digit 'b' in decimal constant}}
__bf16 q = 1bf166; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid digit 'b' in decimal constant}}
__bf16 r = 1bf1; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid digit 'b' in decimal constant}}

__bf16 s = 1bf16; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid digit 'b' in decimal constant}}
__bf16 t = 1BF16; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid digit 'B' in decimal constant}}

__bf16 u = 1.bf16F16; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid suffix 'bf16F16' on floating constant}}
__bf16 v = 1.BF16f16; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid suffix 'BF16f16' on floating constant}}
__bf16 w = 1.F16bf16; // expected-error{{__bf16 is not supported on this target}} expected-error{{invalid suffix 'F16bf16' on floating constant}}

#endif
