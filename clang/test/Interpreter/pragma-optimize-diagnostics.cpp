// REQUIRES: host-supports-jit
//
// RUN: cat %s | clang-repl -Xcc -Xclang -Xcc -verify

#pragma clang repl optimize(O0)
#pragma clang repl optimize(O1)
#pragma clang repl optimize(O2)
#pragma clang repl optimize(O3)
#pragma clang repl optimize(Os)
#pragma clang repl optimize(Oz)

#pragma clang repl optimise // expected-warning {{expected identifier in '#pragma clang repl' - ignored}}
#pragma clang repl optimize // expected-warning {{missing '(' after '#pragma clang repl optimize' - ignoring}}
#pragma clang repl optimize(2) // expected-warning {{unexpected argument '2' to '#pragma clang repl optimize'; expected 'O0', 'O1', 'O2', 'O3', 'Os' or 'Oz'}}
#pragma clang repl optimize(foo) // expected-warning {{unexpected argument 'foo' to '#pragma clang repl optimize'}}
#pragma clang repl optimize(O4) // expected-warning {{unexpected argument 'O4' to '#pragma clang repl optimize'}}
#pragma clang repl optimize(Og) // expected-warning {{unexpected argument 'Og' to '#pragma clang repl optimize'}}
#pragma clang repl optimize(Ofast) // expected-warning {{unexpected argument 'Ofast' to '#pragma clang repl optimize'}}
#pragma clang repl optimize(O2 // expected-warning {{missing ')' after '#pragma clang repl optimize' - ignoring}}
#pragma clang repl optimize(O2) extra // expected-warning {{extra tokens at end of '#pragma clang repl optimize' - ignored}}
