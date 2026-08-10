// RUN: clang-tblgen -gen-clang-opencl-builtin-tests %clang_src_sema_dir/OpenCLBuiltins.td -o %t.cl
// RUN: %clang_cc1 -include %s %t.cl -triple spir -verify -fsyntax-only -cl-std=CL2.0 -finclude-default-header
// RUN: %clang_cc1 -include %s %t.cl -triple spir -verify -fsyntax-only -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header

// Generate an OpenCL source file containing a call to each builtin from
// OpenCLBuiltins.td and then run that generated source file through the
// frontend.
//
// Then test that:
//  - The generated file can be parsed using opencl-c.h, giving some confidence
//    that OpenCLBuiltins.td does not provide more than what opencl-c.h provides
//    (but not vice versa).
//
//  - The generated file can be parsed using -fdeclare-opencl-builtins, ensuring
//    some internal consistency of declarations in OpenCLBuiltins.td.  For
//    example, addition of builtin declarations that lead to ambiguity during
//    overload resolution will cause this test to fail.

// expected-no-diagnostics
