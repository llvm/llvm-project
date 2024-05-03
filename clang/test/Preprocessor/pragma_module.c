// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'module foo { module a {} module b {} } module bar {} module if {}' > %t/module.modulemap
// RUN: %clang_cc1 -fmodules -fmodule-name=if -x c %t/module.modulemap -emit-module -o %t/if.pcm
// RUN: %clang_cc1 -E -fmodules %s -fmodule-file=%t/if.pcm -verify -fmodule-name=foo -fmodule-map-file=%t/module.modulemap
// RUN: %clang_cc1 -E -fmodules %s -fmodule-file=%t/if.pcm -verify -fmodule-name=foo -fmodule-map-file=%t/module.modulemap -fmodules-local-submodule-visibility -DLOCAL_VIS

// Just checking the syntax here; the semantics are tested elsewhere.
#pragma clang module import // expected-error {{expected module name}}
#pragma clang module import ! // expected-error {{expected module name}}
#pragma clang module import if // ok
#pragma clang module import foo ? bar // expected-warning {{extra tokens at end of #pragma}}
#pragma clang module import foo. // expected-error {{expected identifier after '.' in module name}}
#pragma clang module import foo.bar.baz.quux // expected-error {{no submodule named 'bar' in module 'foo'}}

#pragma clang module begin ! // expected-error {{expected module name}}

#pragma clang module begin foo.a blah // expected-warning {{extra tokens}}
 #pragma clang module begin foo.a // nesting is OK
  #define X 1 // expected-note 0-1{{previous}}
  #ifndef X
  #error X should be defined here
  #endif
 #pragma clang module end
 
 #ifndef X
 #error X should still be defined
 #endif
#pragma clang module end foo.a // expected-warning {{extra tokens}}

// #pragma clang module begin/end also import the module into the enclosing context
#ifndef X
#error X should still be defined
#endif

#pragma clang module begin foo.b
 #if defined(X) && defined(LOCAL_VIS)
 #error under -fmodules-local-submodule-visibility, X should not be defined
 #endif

 #if !defined(X) && !defined(LOCAL_VIS)
 #error without -fmodules-local-submodule-visibility, X should still be defined
 #endif

 #pragma clang module import foo.a
 #ifndef X
 #error X should be defined here
 #endif
#pragma clang module end

#pragma clang module end // expected-error {{no matching '#pragma clang module begin'}}
#pragma clang module begin foo.a // expected-error {{no matching '#pragma clang module end'}}
