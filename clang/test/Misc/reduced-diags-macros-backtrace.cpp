// RUN: not %clang_cc1 -fsyntax-only -fmacro-backtrace-limit=0 %s 2>&1 | FileCheck %s --check-prefix=ALL
// RUN: not %clang_cc1 -fsyntax-only -fmacro-backtrace-limit=2 %s 2>&1 | FileCheck %s --check-prefix=SKIP

#define F(x) x + 1
#define G(x) F(x) + 2
#define ADD(x,y) G(x) + y
#define LEVEL4(x) ADD(p,x)
#define LEVEL3(x) LEVEL4(x)
#define LEVEL2(x) LEVEL3(x)
#define LEVEL1(x) LEVEL2(x)

int a = LEVEL1(b);

// ALL: {{.*}}:12:9: error: use of undeclared identifier 'p'
// ALL-NEXT: int a = LEVEL1(b);
// ALL-NEXT:         ^
// ALL-NEXT: {{.*}}:10:19: note: expanded from macro 'LEVEL1'
// ALL-NEXT: #define LEVEL1(x) LEVEL2(x)
// ALL-NEXT:                   ^
// ALL-NEXT: {{.*}}:9:19: note: expanded from macro 'LEVEL2'
// ALL-NEXT: #define LEVEL2(x) LEVEL3(x)
// ALL-NEXT:                   ^
// ALL-NEXT: {{.*}}:8:19: note: expanded from macro 'LEVEL3'
// ALL-NEXT: #define LEVEL3(x) LEVEL4(x)
// ALL-NEXT:                   ^
// ALL-NEXT: {{.*}}:7:23: note: expanded from macro 'LEVEL4'
// ALL-NEXT: #define LEVEL4(x) ADD(p,x)
// ALL-NEXT:                       ^
// ALL-NEXT: {{.*}}:12:16: error: use of undeclared identifier 'b'
// ALL-NEXT: int a = LEVEL1(b);
// ALL-NEXT:                ^
// ALL-NEXT: 2 errors generated.

// SKIP: {{.*}}:12:9: error: use of undeclared identifier 'p'
// SKIP-NEXT: int a = LEVEL1(b);
// SKIP-NEXT:         ^
// SKIP-NEXT: {{.*}}:10:19: note: expanded from macro 'LEVEL1'
// SKIP-NEXT: #define LEVEL1(x) LEVEL2(x)
// SKIP-NEXT:                   ^
// SKIP-NEXT: note: (skipping 2 expansions in backtrace; use -fmacro-backtrace-limit=0 to see all)
// SKIP-NEXT: {{.*}}:7:23: note: expanded from macro 'LEVEL4'
// SKIP-NEXT: #define LEVEL4(x) ADD(p,x)
// SKIP-NEXT:                       ^
// SKIP-NEXT: {{.*}}:12:16: error: use of undeclared identifier 'b'
// SKIP-NEXT: int a = LEVEL1(b);
// SKIP-NEXT:                ^
// SKIP-NEXT: 2 errors generated.
