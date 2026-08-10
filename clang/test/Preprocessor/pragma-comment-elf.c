// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-extensions -triple x86_64-scei-ps4
// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-extensions -triple x86_64-sie-ps5
// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-extensions -triple i686-unknown-linux-gnu

// On ELF targets, issue a diagnostic that pragma comments are ignored except:
//   #pragma comment lib

#pragma comment(lib)
#pragma comment(lib,"foo")
__pragma(comment(lib, "bar"))

#pragma comment(linker) // expected-warning {{'#pragma comment linker' ignored}}
#pragma comment(linker,"foo") // expected-warning {{'#pragma comment linker' ignored}}
__pragma(comment(linker, " bar=" "2")) // expected-warning {{'#pragma comment linker' ignored}}

#pragma comment(user) // expected-warning {{'#pragma comment user' ignored}} 
#pragma comment(user, "Compiled on " __DATE__ " at " __TIME__ ) // expected-warning {{'#pragma comment user' ignored}}
__pragma(comment(user, "foo")) // expected-warning {{'#pragma comment user' ignored}}

#pragma comment(compiler) // expected-warning {{'#pragma comment compiler' ignored}}
#pragma comment(compiler, "foo") // expected-warning {{'#pragma comment compiler' ignored}}
__pragma(comment(compiler, "foo")) // expected-warning {{'#pragma comment compiler' ignored}}

#pragma comment(exestr) // expected-warning {{'#pragma comment exestr' ignored}}
#pragma comment(exestr, "foo") // expected-warning {{'#pragma comment exestr' ignored}}
__pragma(comment(exestr, "foo")) // expected-warning {{'#pragma comment exestr' ignored}}

#pragma comment(foo)    // expected-error {{unknown kind of pragma comment}}
__pragma(comment(foo))  // expected-error {{unknown kind of pragma comment}}
