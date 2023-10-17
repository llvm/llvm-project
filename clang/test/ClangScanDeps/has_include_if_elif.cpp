// RUN: rm -rf %t
// RUN: split-file %s %t

//--- cdb.json.in
[{
  "directory": "DIR",
  "command": "clang -c DIR/tu.c -o DIR/tu.o -IDIR/include",
  "file": "DIR/tu.c"
}]
//--- include/header1.h
//--- include/header2.h
//--- include/header3.h
//--- include/header4.h
//--- tu.c
#if __has_include("header1.h")
#endif

#if 0
#elif __has_include("header2.h")
#endif

#define H3 __has_include("header3.h")
#if H3
#endif

#define H4 __has_include("header4.h")
#if 0
#elif H4
#endif

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -mode preprocess-dependency-directives | FileCheck %s
// RUN: clang-scan-deps -compilation-database %t/cdb.json -mode preprocess | FileCheck %s

// CHECK: tu.c
// CHECK-NEXT: header1.h
// CHECK-NEXT: header2.h
// CHECK-NEXT: header3.h
// CHECK-NEXT: header4.h
