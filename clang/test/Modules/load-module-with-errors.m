// Note: the run lines follow their respective tests, since line/column
// matter in this test.

// pcherror-error@* {{PCH file contains compiler errors}}
@import use_error_a; // notallowerror-error {{could not build module 'use_error_a'}}
@import use_error_b;
// expected-no-diagnostics

void test(Error *x) {
  funca(x);
  funcb(x);
  [x method];
}

// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: mkdir %t/prebuilt

// RUN: %clang_cc1 -fmodules -fallow-pcm-with-compiler-errors \
// RUN:   -fmodule-name=error -o %t/prebuilt/error.pcm \
// RUN:   -x objective-c -emit-module %S/Inputs/error/module.modulemap
// RUN: %clang_cc1 -fmodules -fallow-pcm-with-compiler-errors \
// RUN:   -fmodule-file=error=%t/prebuilt/error.pcm \
// RUN:   -fmodule-name=use_error_a -o %t/prebuilt/use_error_a.pcm \
// RUN:   -x objective-c -emit-module %S/Inputs/error/module.modulemap
// RUN: %clang_cc1 -fmodules -fallow-pcm-with-compiler-errors \
// RUN:   -fmodule-file=error=%t/prebuilt/error.pcm \
// RUN:   -fmodule-name=use_error_b -o %t/prebuilt/use_error_b.pcm \
// RUN:   -x objective-c -emit-module %S/Inputs/error/module.modulemap

// Prebuilt modules
// RUN: %clang_cc1 -fmodules -fallow-pcm-with-compiler-errors \
// RUN:   -fprebuilt-module-path=%t/prebuilt -fmodules-cache-path=%t \
// RUN:   -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fmodules \
// RUN:   -fprebuilt-module-path=%t/prebuilt -fmodules-cache-path=%t \
// RUN:   -verify=pcherror %s

// Explicit prebuilt modules (loaded when needed)
// RUN: %clang_cc1 -fmodules -fallow-pcm-with-compiler-errors \
// RUN:   -fmodule-file=error=%t/prebuilt/error.pcm \
// RUN:   -fmodule-file=use_error_a=%t/prebuilt/use_error_a.pcm \
// RUN:   -fmodule-file=use_error_b=%t/prebuilt/use_error_b.pcm \
// RUN:   -fmodules-cache-path=%t -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fmodules \
// RUN:   -fmodule-file=error=%t/prebuilt/error.pcm \
// RUN:   -fmodule-file=use_error_a=%t/prebuilt/use_error_a.pcm \
// RUN:   -fmodule-file=use_error_b=%t/prebuilt/use_error_b.pcm \
// RUN:   -fmodules-cache-path=%t -verify=pcherror %s

// Explicit prebuilt modules without name (always loaded)
// RUN: %clang_cc1 -fmodules -fallow-pcm-with-compiler-errors \
// RUN:   -fmodule-file=%t/prebuilt/error.pcm \
// RUN:   -fmodule-file=%t/prebuilt/use_error_a.pcm \
// RUN:   -fmodule-file=%t/prebuilt/use_error_b.pcm \
// RUN:   -fmodules-cache-path=%t -ast-print %s | FileCheck %s
// As the modules are always loaded, compiling will fail before even parsing
// this file - this means that -verify can't be used, so do a grep instead.
// RUN: not %clang_cc1 -fsyntax-only -fmodules \
// RUN:   -fmodule-file=%t/prebuilt/error.pcm \
// RUN:   -fmodule-file=%t/prebuilt/use_error_a.pcm \
// RUN:   -fmodule-file=%t/prebuilt/use_error_b.pcm \
// RUN:   -fmodules-cache-path=%t 2>&1 | \
// RUN: grep "PCH file contains compiler errors"

// Shouldn't build the cached modules (that have errors) when not allowing
// errors
// RUN: not %clang_cc1 -fsyntax-only -fmodules \
// RUN:   -fmodules-cache-path=%t -fimplicit-module-maps -I %S/Inputs/error \
// RUN:   -x objective-c %s
// RUN: find %t -name "error-*.pcm" | not grep error

// Should build the cached modules when allowing errors
// RUN: %clang_cc1 -fsyntax-only -fmodules -fallow-pcm-with-compiler-errors \
// RUN:   -fmodules-cache-path=%t -fimplicit-module-maps -I %S/Inputs/error \
// RUN:   -x objective-c -verify %s
// RUN: find %t -name "error-*.pcm" | grep error
// RUN: find %t -name "use_error_a-*.pcm" | grep use_error_a
// RUN: find %t -name "use_error_b-*.pcm" | grep use_error_b

// Check build when the modules are already cached
// RUN: %clang_cc1 -fsyntax-only -fmodules -fallow-pcm-with-compiler-errors \
// RUN:   -fmodules-cache-path=%t -fimplicit-module-maps -I %S/Inputs/error \
// RUN:   -x objective-c -verify %s

// Should rebuild the cached module if it had an error (if it wasn't rebuilt
// the verify would fail as it would be the PCH error instead)
// RUN: %clang_cc1 -fsyntax-only -fmodules \
// RUN:   -fmodules-cache-path=%t -fimplicit-module-maps -I %S/Inputs/error \
// RUN:   -x objective-c %s -verify=notallowerror

// allow-pcm-with-compiler-errors should also allow errors in PCH
// RUN: %clang_cc1 -fallow-pcm-with-compiler-errors -x objective-c \
// RUN:   -o %t/check.pch -emit-pch %S/Inputs/error/error.h

// CHECK: @interface Error
// CHECK-NEXT: - (int)method;
// CHECK-NEXT: - (id)method2;
// CHECK-NEXT: @end
// CHECK: void test(Error *x)

// RUN: c-index-test -code-completion-at=%s:12:6 %s -fmodules -fmodules-cache-path=%t \
// RUN:   -Xclang -fallow-pcm-with-compiler-errors -I %S/Inputs/error | FileCheck -check-prefix=COMPLETE %s
// COMPLETE: ObjCInstanceMethodDecl:{ResultType int}{TypedText method}
// COMPLETE: ObjCInstanceMethodDecl:{ResultType id}{TypedText method2}

// RUN: c-index-test -test-load-source local %s -fmodules -fmodules-cache-path=%t \
// RUN:   -Xclang -fallow-pcm-with-compiler-errors -I %S/Inputs/error | FileCheck -check-prefix=SOURCE %s
// SOURCE: load-module-with-errors.m:9:6: FunctionDecl=test:9:6 (Definition) Extent=[9:1 - 13:2]
// SOURCE: load-module-with-errors.m:9:18: ParmDecl=x:9:18 (Definition) Extent=[9:11 - 9:19]
// SOURCE: load-module-with-errors.m:9:11: ObjCClassRef=Error:5:12 Extent=[9:11 - 9:16]
// SOURCE: load-module-with-errors.m:9:21: CompoundStmt= Extent=[9:21 - 13:2]
// SOURCE: load-module-with-errors.m:10:3: CallExpr=funca:3:6 Extent=[10:3 - 10:11]
// SOURCE: load-module-with-errors.m:11:3: CallExpr=funcb:3:6 Extent=[11:3 - 11:11]
// SOURCE: load-module-with-errors.m:12:3: ObjCMessageExpr=method:6:8 Extent=[12:3 - 12:13]
