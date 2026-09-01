// Locations in a file reused from an earlier module must survive a
// serialize/deserialize round trip. dup.h is written into q.pcm and reused when
// a.pcm loads it, so DECL(Box) gives Box macro expansion locations inside that
// reused file. Writing b.pcm re-encodes those loaded locations, which requires
// inverting the reader's local-to-global map. While that inverse assumed a flat
// shift it did not invert the piecewise map, and writing b.pcm asserted while
// serializing 'Box<int>'.
//
// Assertions are required: with the inverse wrong but assertions off, every
// step below still succeeds.

// REQUIRES: asserts

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/q.h \
// RUN:   -I%t -Wno-experimental-header-units -o %t/q.pcm
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/a.h \
// RUN:   -I%t -fmodule-file=%t/q.pcm -Wno-experimental-header-units -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/b.h \
// RUN:   -I%t -fmodule-file=%t/a.pcm -Wno-experimental-header-units -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only %t/use.cpp -I%t \
// RUN:   -fmodule-file=%t/a.pcm -fmodule-file=%t/b.pcm \
// RUN:   -Wno-experimental-header-units

//--- dup.h
#define DECL(name) template <class T> struct name { T value; };

//--- q.h
#include "dup.h"

//--- a.h
import "q.h";
#include "dup.h"
DECL(Box)

//--- b.h
import "a.h";
using Alias = Box<int>;

//--- use.cpp
import "a.h";
import "b.h";
Alias value;
int main() { return value.value; }
