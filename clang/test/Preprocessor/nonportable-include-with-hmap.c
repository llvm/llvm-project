// REQUIRES: shell
// REQUIRES: case-insensitive-filesystem

// RUN: rm -f %t.hmap
// RUN: sed -e "s:INPUTS_DIR:%S/Inputs:g" \
// RUN:   %S/Inputs/nonportable-hmaps/foo.hmap.json > %t.hmap.json
// RUN: %hmaptool write %t.hmap.json %t.hmap
// RUN: %clang_cc1 -Eonly                        \
// RUN:   -I%t.hmap \
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
#include "Foo/Foo.h" // no warning

// Verify files with absolute paths in the header map are handled too.
// "Bar.h" is included twice to make sure that when we see potentially
// nonportable path, the file has been already discovered through a relative
// path which triggers the file to be opened and `FileEntry::RealPathName`
// to be set.
#include "Bar.h"
#include "Foo/Bar.h" // no warning

// But the presence of the absolute path in the header map is not enough. If we
// didn't use it to discover a file, shouldn't suppress the warning.
#include "headers/Foo/Baz.h" // expected-warning {{non-portable path}}
