// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json

//--- cdb.json.in
[{
    "directory": "DIR",
    "command": "clang -c DIR/tu.c -o DIR/tu.o -IDIR/include",
    "file": "DIR/tu.c"
}]
//--- include/header.h
//--- include/header2.h
//--- include/header3.h
//--- include/header4.h
//--- include/header5.h
//--- include/header6.h
//--- include/header7.h
//--- include/header8.h
//--- include/header9.h
//--- include/header10.h
//--- include/header11.h
//--- include/header12.h
//--- include/header13.h
//--- include/header14.h
//--- tu.c
#if 0
R"x()x"
#endif

#include "header.h"

#if 0
R"y(";
#endif
#include "header2.h"

#if 0
//")y"
#endif

#if 0
R"y(";
R"z()y";
#endif
#include "header3.h"
#if 0
//")z"
#endif

#if 0
R\
"y(";
R"z()y";
#endif
#include "header4.h"
#if 0
//")z"
#endif

// Test u8 prefix with escaped newline
#if 0
u8R\
"prefix(test)prefix"
#endif
#include "header5.h"

// Test u prefix with multiple escaped newlines
#if 0
uR\
\
"multi(test)multi"
#endif
#include "header6.h"

// Test U prefix with escaped newline
#if 0
UR\
"upper(test)upper"
#endif
#include "header7.h"

// Test L prefix with escaped newline
#if 0
LR\
"wide(test)wide"
#endif
#include "header8.h"

// Test escaped newline with \r\n style
#if 0
R\
"crlf(test)crlf"
#endif
#include "header9.h"

// Test multiple escaped newlines in different positions
#if 0
u\
8\
R\
"complex(test)complex"
#endif
#include "header10.h"

// Test raw string that should NOT be treated as raw (no R prefix due to identifier continuation)
#if 0
identifierR"notraw(test)notraw"
#endif
#include "header11.h"

// Test raw string with whitespace before escaped newline
#if 0
R \
"whitespace(test)whitespace"
#endif
#include "header12.h"

// Test nested raw strings in disabled code
#if 0
R"outer(
    R"inner(content)inner"
)outer"
#endif
#include "header13.h"

// Test raw string with empty delimiter
#if 0
R\
"(empty delimiter)";
#endif
#include "header14.h"

// RUN: clang-scan-deps -compilation-database %t/cdb.json -mode preprocess | FileCheck %s
// RUN: clang-scan-deps -compilation-database %t/cdb.json -mode preprocess-dependency-directives | FileCheck %s
// CHECK: tu.c
// CHECK-NEXT: header.h
// CHECK-NEXT: header3.h
// CHECK-NEXT: header4.h
// CHECK-NEXT: header5.h
// CHECK-NEXT: header6.h
// CHECK-NEXT: header7.h
// CHECK-NEXT: header8.h
// CHECK-NEXT: header9.h
// CHECK-NEXT: header10.h
// CHECK-NEXT: header11.h
// CHECK-NEXT: header12.h
// CHECK-NEXT: header13.h
// CHECK-NEXT: header14.h
