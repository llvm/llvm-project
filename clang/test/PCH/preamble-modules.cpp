// Check that modules included in the preamble remain visible to the rest of the
// file.

// RUN: rm -rf %t.mcp
// RUN: %clang_cc1 -emit-pch -o %t.pch %s -fmodules -fmodule-map-file=%S/Inputs/modules/module.modulemap -fmodules-local-submodule-visibility -fmodules-cache-path=%t.mcp
// RUN: %clang_cc1 -include-pch %t.pch %s -fmodules -fmodule-map-file=%S/Inputs/modules/module.modulemap -fmodules-local-submodule-visibility -fmodules-cache-path=%t.mcp

#ifndef MAIN_FILE
#define MAIN_FILE
// Premable section.
#include "Inputs/modules/Foo.h"
#else
// Main section.
MyType foo;
#endif
