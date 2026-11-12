// This test demonstrates how -fmodule-map-file-home-is-cwd with -fmodules-embed-all-files
// extend the importer search paths by relying on the side effects of pragma diagnostic
// mappings deserialization.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- dir1/a.modulemap
module a { header "a.h" }
//--- dir1/a.h
#include "search.h"
// The first compilation is configured such that -I search does contain the search.h header.
//--- dir1/search/search.h
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wparentheses"
#pragma clang diagnostic pop
// RUN: cd %t/dir1 && %clang_cc1 -fmodules -I search \
// RUN:   -emit-module -fmodule-name=a a.modulemap -o %t/a.pcm \
// RUN:   -fmodules-embed-all-files -fmodule-map-file-home-is-cwd

//--- dir2/b.modulemap
module b { header "b.h" }
//--- dir2/b.h
#include "search.h" // expected-error{{'search.h' file not found}}
// The second compilation is configured such that -I search is an empty directory.
// However, since b.pcm simply embeds the headers as "search/search.h", this compilation
// ends up seeing it too. This relies solely on ASTReader::ReadPragmaDiagnosticMappings()
// eagerly reading the corresponding INPUT_FILE record before header search happens.
// Removing the eager deserialization makes this header invisible and so does removing
// the pragma directives.
// RUN: mkdir %t/dir2/search
// RUN: cd %t/dir2 && %clang_cc1 -fmodules -I search \
// RUN:   -emit-module -fmodule-name=b b.modulemap -o %t/b.pcm \
// RUN:   -fmodule-file=%t/a.pcm -verify
