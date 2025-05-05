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

// RUN: clang-scan-deps -compilation-database %t/cdb.json -mode preprocess | FileCheck %s
// RUN: clang-scan-deps -compilation-database %t/cdb.json -mode preprocess-dependency-directives | FileCheck %s
// CHECK: tu.c
// CHECK-NEXT: header.h
// CHECK-NEXT: header3.h
// CHECK-NEXT: header4.h
