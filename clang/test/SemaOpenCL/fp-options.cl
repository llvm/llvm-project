// RUN: %clang_cc1 %s -finclude-default-header -fdeclare-opencl-builtins -triple spir-unknown-unknown -emit-pch -o %t.pch
// RUN: %clang_cc1 %s -finclude-default-header -fdeclare-opencl-builtins -cl-no-signed-zeros -triple spir-unknown-unknown -include-pch %t.pch -fsyntax-only -verify
// expected-no-diagnostics
