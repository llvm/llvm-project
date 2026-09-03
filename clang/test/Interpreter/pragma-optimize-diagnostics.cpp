// REQUIRES: host-supports-jit
//
// RUN: cat %s | clang-repl -Xcc -Xclang -Xcc -verify

#pragma clang repl optimise // expected-warning {{expected identifier in '#pragma clang repl' - ignored}}
#pragma clang repl optimize // expected-warning {{missing '(' after '#pragma clang repl optimize' - ignoring}}
#pragma clang repl optimize(2) // expected-warning {{unexpected argument '2' to '#pragma clang repl optimize'; expected an optimization flag such as 'O2' or 'Os'}}
#pragma clang repl optimize(foo) // expected-warning {{unexpected argument 'foo' to '#pragma clang repl optimize'; expected an optimization flag such as 'O2' or 'Os'}}
#pragma clang repl optimize(Ofast) // expected-warning {{known but unsupported action 'Ofast' for '#pragma clang repl optimize' - ignored}}
#pragma clang repl optimize(O2 // expected-warning {{missing ')' after '#pragma clang repl optimize' - ignoring}}
#pragma clang repl optimize(O2) extra // expected-warning {{extra tokens at end of '#pragma clang repl optimize' - ignored}}
