// RUN: rm -rf %t
// Build A.pcm
// RUN: %clang_cc1 -Werror -Wno-conversion -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs %s -fmodules-disable-diagnostic-validation
// Build pch that imports A.pcm
// RUN: %clang_cc1 -Werror -Wno-conversion -emit-pch -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -o %t.pch -I %S/Inputs -x objective-c-header %S/Inputs/pch-import-module-out-of-date.pch -fmodules-disable-diagnostic-validation
// Make sure that we don't rebuild A.pcm and overwrite the original A.pcm that the pch imports
// RUN: %clang_cc1 -Werror -Wconversion -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs %s -fmodules-disable-diagnostic-validation
// Make sure we don't error out when using the pch
// RUN: %clang_cc1 -Werror -Wno-conversion -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -fsyntax-only -I %S/Inputs -include-pch %t.pch %s -verify -fmodules-disable-diagnostic-validation

// Build A.pcm
// RUN: %clang_cc1 -Werror -Wno-conversion -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs %s
// Build pch that imports A.pcm
// RUN: %clang_cc1 -Werror -Wno-conversion -emit-pch -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -o %t.pch -I %S/Inputs -x objective-c-header %S/Inputs/pch-import-module-out-of-date.pch
// We will rebuild A.pcm and overwrite the original A.pcm that the pch imports, but the two versions have the same hash.
// RUN: %clang_cc1 -Werror -Wconversion -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs %s
// Make sure we don't error out when using the pch
// RUN: %clang_cc1 -Werror -Wno-conversion -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -fsyntax-only -I %S/Inputs -include-pch %t.pch %s -verify

// expected-no-diagnostics

@import DiagOutOfDate;
