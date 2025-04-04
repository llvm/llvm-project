// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s
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
