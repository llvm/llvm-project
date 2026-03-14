// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fmodules-decluse -fmodule-name=A -fmodule-map-file=%S/Inputs/separate_map_tree/maps/modulea.map -I %S/Inputs/separate_map_tree/src %s -verify

#include "common.h"
#include "public-in-b.h" // expected-error {{private header}}
#include "public-in-c.h"
#include "private-in-c.h" // expected-error {{private header}}
const int val = common + b + c + c_;
