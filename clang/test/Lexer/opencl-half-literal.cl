// RUN: %clang_cc1 %s -fsyntax-only -verify -triple spir-unknown-unknown

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

constant half a = 1.0h; 
constant half aa = 1.0H;
constant half b = 1.0hh; // expected-error{{invalid suffix 'hh' on floating constant}}
constant half c = 1.0fh; // expected-error{{invalid suffix 'fh' on floating constant}}
constant half d = 1.0lh; // expected-error{{invalid suffix 'lh' on floating constant}}
constant half e = 1.0hf; // expected-error{{invalid suffix 'hf' on floating constant}}
