// Check how builtins using varargs behave with the modules.

// REQUIRES: x86-registered-target
// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -triple x86_64-apple-darwin \
// RUN:   -fmodules -fno-implicit-modules -fbuiltin-headers-in-system-modules \
// RUN:   -emit-module -fmodule-name=DeclareVarargs \
// RUN:   -x c %t/include/module.modulemap -o %t/DeclareVarargs.pcm \
// RUN:   -fmodule-map-file=%t/resource_dir/module.modulemap -isystem %t/resource_dir
// RUN: %clang_cc1 -triple x86_64-apple-darwin \
// RUN:   -fmodules -fno-implicit-modules -fbuiltin-headers-in-system-modules \
// RUN:   -emit-pch -fmodule-name=Prefix \
// RUN:   -x c-header %t/prefix.pch -o %t/prefix.pch.gch \
// RUN:   -fmodule-map-file=%t/include/module.modulemap -fmodule-file=DeclareVarargs=%t/DeclareVarargs.pcm \
// RUN:   -I %t/include
// RUN: %clang_cc1 -triple x86_64-apple-darwin \
// RUN:   -fmodules -fno-implicit-modules -fbuiltin-headers-in-system-modules \
// RUN:   -emit-obj -fmodule-name=test \
// RUN:   -x c %t/test.c -o %t/test.o \
// RUN:   -Werror=incompatible-pointer-types \
// RUN:   -fmodule-file=%t/DeclareVarargs.pcm -include-pch %t/prefix.pch.gch \
// RUN:   -I %t/include

//--- include/declare-varargs.h
#ifndef DECLARE_VARARGS_H
#define DECLARE_VARARGS_H

#include <stdarg.h>

int vprintf(const char *format, va_list args);

// 1. initializeBuiltins 'acos' causes its deserialization and deserialization
//    of 'implementation_of_builtin'. Because this is done before Sema initialization,
//    'implementation_of_builtin' DeclID is added to PreloadedDeclIDs.
#undef acos
#define acos(__x) implementation_of_builtin(__x)

// 2. Because of 'static' the function isn't added to EagerlyDeserializedDecls
//    and not deserialized in `ASTReader::StartTranslationUnit` before `ASTReader::InitializeSema`.
// 3. Because of '__overloadable__' attribute the function requires name mangling during deserialization.
//    And the name mangling requires '__builtin_va_list' decl.
//    Because the function is added to PreloadedDeclIDs, the deserialization happens in `ASTReader::InitializeSema`.
static int __attribute__((__overloadable__)) implementation_of_builtin(int x) {
  return x;
}

#endif // DECLARE_VARARGS_H

//--- include/module.modulemap
module DeclareVarargs {
  header "declare-varargs.h"
  export *
}

//--- resource_dir/stdarg.h
#ifndef STDARG_H
#define STDARG_H

typedef __builtin_va_list va_list;
#define va_start(ap, param) __builtin_va_start(ap, param)
#define va_end(ap) __builtin_va_end(ap)

#endif // STDARG_H

//--- resource_dir/module.modulemap
module _Builtin_stdarg {
  header "stdarg.h"
  export *
}

//--- prefix.pch
#include <declare-varargs.h>

//--- test.c
#include <declare-varargs.h>

void test(const char *format, ...) {
  va_list argParams;
  va_start(argParams, format);
  vprintf(format, argParams);
  va_end(argParams);
}
