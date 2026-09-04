// Tests for --mapping-file support (IWYU .imp format).
// umbrella_inc.h transitively provides private_func via private_header.h.

// Set up a temporary include-mapping file: private_header.h -> <public_header.h>
// RUN: echo '[{"include": ["<private_header.h>", "private", "<public_header.h>", "public"]}]' > %t.inc.imp

// Without a mapping file: the tool suggests the direct provider (private_header.h).
// RUN: clang-include-cleaner -print=changes %s -- -I%S/Inputs/ | \
// RUN:   FileCheck --check-prefix=NOMAP %s
// NOMAP: - "umbrella_inc.h"
// NOMAP: + "private_header.h"
// NOMAP-NOT: public_header

// With the include mapping file: the tool suggests the mapped public header.
// RUN: clang-include-cleaner --mapping-file=%t.inc.imp -print=changes %s -- -I%S/Inputs/ | \
// RUN:   FileCheck --check-prefix=INCMAP %s
// INCMAP: + <public_header.h>
// INCMAP-NOT: + "private_header.h"

// Symbol mapping: map "private_func" symbol to <public_header.h>
// RUN: echo '[{"symbol": ["private_func", "private", "<public_header.h>", "public"]}]' > %t.sym.imp

// With the symbol mapping file: the tool suggests the mapped header for the symbol.
// RUN: clang-include-cleaner --mapping-file=%t.sym.imp -print=changes %s -- -I%S/Inputs/ | \
// RUN:   FileCheck --check-prefix=SYMMAP %s
// SYMMAP: + <public_header.h>

// Multiple mapping files can be specified simultaneously.
// RUN: clang-include-cleaner --mapping-file=%t.inc.imp --mapping-file=%t.sym.imp \
// RUN:   -print=changes %s -- -I%S/Inputs/ | \
// RUN:   FileCheck --check-prefix=MULTI %s
// MULTI: + <public_header.h>

// Regex pattern: "@<private_header.h>" matches private_header.h by suffix.
// RUN: echo '[{"include": ["@<private_header.h>", "private", "<public_header.h>", "public"]}]' > %t.regex.imp
// RUN: clang-include-cleaner --mapping-file=%t.regex.imp -print=changes %s -- -I%S/Inputs/ | \
// RUN:   FileCheck --check-prefix=REGEXMAP %s
// REGEXMAP: + <public_header.h>
// REGEXMAP-NOT: + "private_header.h"

#include "umbrella_inc.h"

int x = private_func();
