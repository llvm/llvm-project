// RUN: rm -rf %t && mkdir %t
// RUN: %hmaptool write %S/Inputs/header-search-crash/foo.hmap.json %t/foo.hmap
// RUN: %clang -cc1 -E %s -I %t/foo.hmap -verify

#include "MissingHeader.h" // expected-error {{'MissingHeader.h' file not found}}
