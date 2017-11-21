// RUN: rm -rf %t
// RUN: mkdir -p %t
//
// RUN: echo 'export module a; export class A{};' | %clang_cc1 -x c++ -fmodules-ts -emit-module-interface - -o %t/a.pcm
// RUN: echo 'export module b; export class B{};' | %clang_cc1 -x c++ -fmodules-ts -emit-module-interface - -o %t/b.pcm
// RUN: echo 'export module c; export class C{};' | %clang_cc1 -x c++ -fmodules-ts -emit-module-interface - -o %t/c.pcm
//
// RUN: %clang_cc1 -fmodules-ts -fprebuilt-module-path=%t -emit-module-interface %s -o %t/aggregate.internal.pcm -DAGGREGATE_INTERNAL
// RUN: %clang_cc1 -fmodules-ts -fprebuilt-module-path=%t -emit-module-interface %s -o %t/aggregate.pcm -DAGGREGATE
//
// RUN: %clang_cc1 -fmodules-ts -fprebuilt-module-path=%t %s -verify -DTEST
// expected-no-diagnostics


#ifdef AGGREGATE_INTERNAL
export module aggregate.internal;
export import a;
export {
  import b;
  import c;
}
#endif


// Export the above aggregate module.
// This is done to ensure that re-exports are transitive.
#ifdef AGGREGATE
export module aggregate;
export import aggregate.internal;
#endif


// For the actual test, just try using the classes from the exported modules
// and hope that they're accessible.
#ifdef TEST
import aggregate;
A a;
B b;
C c;
#endif
