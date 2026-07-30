// RUN: rm -f %t.hmap
// RUN: %hmaptool write %S/Inputs/headermap-rel/foo.hmap.json %t.hmap
// RUN: %clang_cc1 -E %s -o %t.i -I %t.hmap -F %S/Inputs/headermap-rel
// RUN: FileCheck %s -input-file %t.i

// CHECK: Foo.h is parsed
// CHECK: Foo.h is parsed

#include "Foo.h"
#include "Foo.h"
