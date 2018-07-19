// RUN: %clang_cc1 -Eonly                        \
// RUN:   -I%S/Inputs/nonportable-hmaps/foo.hmap \
// RUN:   -I%S/Inputs/nonportable-hmaps          \
// RUN:   %s -verify
//
// foo.hmap contains: Foo/Foo.h -> headers/foo/Foo.h
//
// Header search of "Foo/Foo.h" follows this path:
//  1. Look for "Foo/Foo.h".
//  2. Find "Foo/Foo.h" in "nonportable-hmaps/foo.hmap".
//  3. Look for "headers/foo/Foo.h".
//  4. Find "headers/foo/Foo.h" in "nonportable-hmaps".
//  5. Return.
//
// There is nothing nonportable; -Wnonportable-include-path should not fire.
#include "Foo/Foo.h" // expected-no-diagnostics
