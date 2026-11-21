// RUN: split-file %s %t
// RUN: %clang_cc1 -E -x assembler-with-cpp %t/config.S | FileCheck %s --strict-whitespace --check-prefix=CONFIG
// RUN: %clang_cc1 -E -traditional-cpp -x assembler-with-cpp %t/config.S | FileCheck %s --strict-whitespace --check-prefix=CONFIG
// RUN: %clang_cc1 -E %t/macros.c | FileCheck %s --strict-whitespace --check-prefix=MACRO
// RUN: %clang_cc1 -E -fminimize-whitespace %t/macros.c | FileCheck %s --strict-whitespace --check-prefix=MACRO
// RUN: %clang_cc1 -E -x assembler-with-cpp %t/macro-arg.S | FileCheck %s --strict-whitespace --check-prefix=ARG
// RUN: %clang_cc1 -E -traditional-cpp -x assembler-with-cpp %t/macro-arg.S | FileCheck %s --strict-whitespace --check-prefix=ARG

//--- config.S
// Hashes originally at the start of a source line must retain their position,
// including when the line comes from an included configuration fragment.
#include "fragment"
# CONFIG_BAR is not set
  # CONFIG_INDENTED is not set
\
# CONFIG_SPLICED is not set
// CONFIG: {{^}}# CONFIG_FOO is not set{{$}}
// CONFIG: {{^}}# CONFIG_BAR is not set{{$}}
// CONFIG: {{^}}  # CONFIG_INDENTED is not set{{$}}
// CONFIG: {{^}}# CONFIG_SPLICED is not set{{$}}

//--- fragment
# CONFIG_FOO is not set

//--- macros.c
// Hashes exposed by removing an empty macro or produced by macro expansion
// must still be indented to avoid becoming directives in -fpreprocessed mode.
#define EMPTY
EMPTY#foo
// MACRO: {{^ +}}#foo{{$}}

#define IDENTITY(x) x
IDENTITY(#) define baz quux
// MACRO: {{^}} #{{ ?}}define baz quux{{$}}

// Unlike EMPTY#foo above, the outer macro puts the hash in column one.
// Propagating StartOfLine from the empty macro must not disable protection.
IDENTITY(EMPTY#wrapped)
// MACRO: {{^}} #wrapped{{$}}

// Real pragmas must remain distinguishable from macro-produced lookalikes,
// including when the hash is written on another line of a macro definition.
#pragma clang diagnostic push
#define PRAGMA \
#pragma clang diagnostic pop
PRAGMA
#pragma clang diagnostic pop
// MACRO: {{^}}#pragma clang diagnostic push{{$}}
// MACRO: {{^}} #pragma clang diagnostic pop{{$}}
// MACRO: {{^}}#pragma clang diagnostic pop{{$}}

//--- macro-arg.S
// The hash retains PhysicalStartOfLine from the argument, but its macro
// location must still prevent it from becoming a directive in the output.
#define ID(x) x
ID(
#) pragma foo
// ARG: {{^}} # pragma foo{{$}}
