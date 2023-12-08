// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:   -F%S/Inputs/incomplete-framework-module \
// RUN:   -fsyntax-only %s -verify

#import <Foo/Foo.h>

// expected-warning@Inputs/incomplete-framework-module/Foo.framework/Modules/module.modulemap:2{{skipping 'Foo.h' because module declaration}}
// expected-warning@Inputs/incomplete-framework-module/Foo.framework/Modules/module.modulemap:3{{skipping 'FooB.h' because module declaration}}
// expected-note@Inputs/incomplete-framework-module/Foo.framework/Modules/module.modulemap:1{{use 'framework module' to declare module 'Foo'}}
// expected-note@Inputs/incomplete-framework-module/Foo.framework/Modules/module.modulemap:1{{use 'framework module' to declare module 'Foo'}}
