// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 -pedantic %s

// Intentionally compiled as C++03 to test the extension warning.

namespace a {} // original
namespace a {} // ext
inline namespace b {} // inline original expected-warning {{inline namespaces are}}
inline namespace b {} // inline ext expected-warning {{inline namespaces are}}
inline namespace {} // inline unnamed expected-warning {{inline namespaces are}}
