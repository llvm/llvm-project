// RUN: %clang_cc1 -E %s 2>&1 | FileCheck %s -strict-whitespace
// PR6101
int a;
// CHECK: # 1 "{{.*}}line-directive-output.c"
// CHECK: int a;

// CHECK-NEXT: # 50 "{{.*}}line-directive-output.c"
// CHECK-NEXT: int b;
#line 50
int b;

// CHECK: # 13 "{{.*}}line-directive-output.c"
// CHECK-NEXT: int c;
# 13
int c;


// CHECK-NEXT: # 1 "A.c"
#line 1 "A.c"
// CHECK-NEXT: # 2 "A.c"
#line 2

// CHECK-NEXT: # 1 "B.c"
#line 1 "B.c"

// CHECK-NEXT: # 1000 "A.c"
#line 1000 "A.c"

int y;







// CHECK: # 1010 "A.c"
int z;

extern int x;

# 3 "temp2.h" 1
extern int y;

# 7 "A.c" 2
extern int z;













// CHECK: # 25 "A.c"


// CHECK: # 50 "C.c" 1
# 50 "C.c" 1


// CHECK-NEXT: # 2000 "A.c" 2
# 2000 "A.c" 2
# 42 "A.c"
# 44 "A.c"
# 49 "A.c"
