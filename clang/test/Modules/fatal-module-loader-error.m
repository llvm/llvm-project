// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: touch %t/Module.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t %s -fdisable-module-hash -F %S/Inputs -verify
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t %s -fdisable-module-hash -F %S/Inputs -DIMPLICIT -verify

// This tests that after a fatal module loader error, we do not continue parsing.

#ifdef IMPLICIT

// expected-error@+1{{Module.pcm' is not a valid precompiled module file}}
#import <Module/Module.h>
#pragma clang __debug crash;

#else

// expected-error@+1{{Module.pcm' is not a valid precompiled module file}}
@import Module;
#pragma clang __debug crash;

#endif

// Also check that libclang does not create a PCH with such an error.
// RUN: not c-index-test -write-pch %t.pch -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN: %s -Xclang -fdisable-module-hash -F %S/Inputs 2>&1 | FileCheck %s
// CHECK: {{^}}Failure: AST deserialization error occurred{{$}}
