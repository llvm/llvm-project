# clang pseudoparser

This directory implements an approximate heuristic parser for C++, based on the
clang lexer, the C++ grammar, and the GLR parsing algorithm.

It parses a file in isolation, without reading its included headers.
The result is a strict syntactic tree whose structure follows the C++ grammar.
There is no semantic analysis, apart from guesses to disambiguate the parse.
Disambiguation can optionally be guided by an AST or a symbol index.

For now, the best reference on intended scope is the [design proposal],
with further discussion on the [RFC].

## Dependencies between pseudoparser and clang

Dependencies are limited because they don't make sense, but also to avoid
placing a burden on clang mantainers.

The pseudoparser reuses the clang lexer (clangLex and clangBasic libraries) but
not the higher-level libraries (Parse, Sema, AST, Frontend...).

When the pseudoparser should be used together with an AST (e.g. to guide
disambiguation), this is a separate "bridge" library that depends on both.

Clang does not depend on the pseudoparser at all. If this seems useful in future
it should be discussed by RFC.

## Parity between pseudoparser and clang

The pseudoparser aims to understand real-world code, and particularly the
languages and extensions supported by Clang.

However we don't try to keep these in lockstep: there's no expectation that
Clang parser changes are accompanied by pseudoparser changes or vice versa.

[design proposal]: https://docs.google.com/document/d/1eGkTOsFja63wsv8v0vd5JdoTonj-NlN3ujGF0T7xDbM/edit
[RFC]: https://discourse.llvm.org/t/rfc-a-c-pseudo-parser-for-tooling/59217/49
