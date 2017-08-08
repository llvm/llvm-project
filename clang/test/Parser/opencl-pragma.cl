// RUN: %clang_cc1 %s -verify -pedantic -Wno-empty-translation-unit -fsyntax-only -triple spir-unknown-unknown

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#pragma OPENCL EXTENSION cl_no_such_extension : disable /* expected-warning {{unknown OpenCL extension 'cl_no_such_extension' - ignoring}} */

#pragma OPENCL EXTENSION all : disable
#pragma OPENCL EXTENSION all : enable /* expected-warning {{expected 'disable' - ignoring}} */

#pragma OPENCL EXTENSION cl_khr_fp64 : on /* expected-warning {{expected 'enable', 'disable', 'begin' or 'end' - ignoring}} */

#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL FP_CONTRACT OFF
#pragma OPENCL FP_CONTRACT DEFAULT
#pragma OPENCL FP_CONTRACT FOO // expected-warning {{expected 'ON' or 'OFF' or 'DEFAULT' in pragma}}
