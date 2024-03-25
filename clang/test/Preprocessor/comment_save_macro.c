// RUN: %clang_cc1 -E -C %s | FileCheck -check-prefix=CHECK-C -strict-whitespace %s
// CHECK-C: boo bork bar // zot
// CHECK-C: ( 0 );
// CHECK-C: ( 0,1,2 );
// CHECK-C: ( 0,1,2 );

// RUN: %clang_cc1 -E -CC %s | FileCheck -check-prefix=CHECK-CC -strict-whitespace %s
// CHECK-CC: boo bork /* blah*/ bar // zot
// CHECK-CC: (/**/0/**/);
// CHECK-CC: (/**/0,1,2/**/);
// CHECK-CC: (/**/0,1,2/**/);

// RUN: %clang_cc1 -E %s | FileCheck -strict-whitespace %s
// CHECK: boo bork bar
// CHECK: ( 0 );
// CHECK: ( 0,1,2 );
// CHECK: ( 0,1,2 );


#define FOO bork // blah
boo FOO bar // zot
#define M(/**/x/**/) (/**/x/**/)
M(0);
#define M2(/**/.../**/) (/**/__VA_ARGS__/**/)
M2(0,1,2);
#define M3(/**/x.../**/) (/**/x/**/)
M3(0,1,2);

