// First trial: pass -DIGNORED=1 to both. This should obviously work.
// RUN: rm -rf %t.modules
// RUN: %clang_cc1 -fmodules-cache-path=%t.modules -DIGNORED=1 -fmodules -fimplicit-module-maps -I %S/Inputs -emit-pch -o %t.pch -x objective-c-header %s -verify
// RUN: %clang_cc1 -fmodules-cache-path=%t.modules -DIGNORED=1 -fmodules -fimplicit-module-maps -I %S/Inputs -include-pch %t.pch %s -verify

// Second trial: pass -DIGNORED=1 only to the second invocation. We
// should detect the failure.
//
// RUN: rm -rf %t.modules
// RUN: %clang_cc1 -fmodules-cache-path=%t.modules -fmodules -fimplicit-module-maps -I %S/Inputs -emit-pch -o %t.pch -x objective-c-header %s -verify
// RUN: not %clang_cc1 -fmodules-cache-path=%t.modules -DIGNORED=1 -fmodules -fimplicit-module-maps -I %S/Inputs -include-pch %t.pch %s > %t.err 2>&1
// RUN: FileCheck -check-prefix=CHECK-CONFLICT %s < %t.err
// CHECK-CONFLICT: PCH was compiled with module cache path

// Third trial: pass -DIGNORED=1 only to the second invocation, but
// make it ignored. There should be no failure, IGNORED is defined in
// the translation unit but not the module.
// RUN: rm -rf %t.modules
// RUN: %clang_cc1 -fmodules-cache-path=%t.modules -fmodules -fimplicit-module-maps -I %S/Inputs -emit-pch -o %t.pch -x objective-c-header %s -verify
// RUN: %clang_cc1 -fmodules-cache-path=%t.modules -DIGNORED=1 -fmodules -fimplicit-module-maps -I %S/Inputs -include-pch %t.pch -fmodules-ignore-macro=IGNORED %s -verify

// Fourth trial: pass -DIGNORED=1 and -fmodules-ignore-macro=IGNORED
// to both invocations, so modules will be built without the IGNORED
// macro.
// RUN: rm -rf %t.modules
// RUN: %clang_cc1 -fmodules-cache-path=%t.modules -DIGNORED=1 -fmodules-ignore-macro=IGNORED -fmodules -fimplicit-module-maps -I %S/Inputs -emit-pch -o %t.pch -x objective-c-header %s -verify
// RUN: %clang_cc1 -fmodules-cache-path=%t.modules -DIGNORED=1 -fmodules -fimplicit-module-maps -I %S/Inputs -include-pch %t.pch -fmodules-ignore-macro=IGNORED -DNO_IGNORED_ANYWHERE -fmodules-ignore-macro=NO_IGNORED_ANYWHERE %s -verify

// Fifth trial: pass -DIGNORED=1 and -fmodules-ignore-macro=IGNORED=1
// to both invocations, so modules will be built without the IGNORED
// macro.
// RUN: rm -rf %t.modules
// RUN: %clang_cc1 -fmodules-cache-path=%t.modules -DIGNORED=1 -fmodules-ignore-macro=IGNORED=1 -fmodules -fimplicit-module-maps -I %S/Inputs -emit-pch -o %t.pch -x objective-c-header %s -verify
// RUN: %clang_cc1 -fmodules-cache-path=%t.modules -DIGNORED=1 -fmodules -fimplicit-module-maps -I %S/Inputs -include-pch %t.pch -fmodules-ignore-macro=IGNORED=1 -DNO_IGNORED_ANYWHERE -fmodules-ignore-macro=NO_IGNORED_ANYWHERE %s -verify

// expected-no-diagnostics

#ifndef HEADER
#define HEADER
@import ignored_macros;
#endif

@import ignored_macros;

struct Point p;

#ifdef NO_IGNORED_ANYWHERE
void *has_ignored(int, int, int);
#endif
