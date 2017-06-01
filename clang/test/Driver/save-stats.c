// RUN: %clang -target x86_64-apple-darwin -save-stats %s -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-apple-darwin -save-stats=cwd %s -### 2>&1 | FileCheck %s
// CHECK: "-stats-file=save-stats.stats"
// CHECK: "{{.*}}save-stats.c"

// RUN: %clang -target x86_64-apple-darwin -S %s -### 2>&1 | FileCheck %s -check-prefix=NO-STATS
// NO-STATS-NO: -stats-file
// NO-STATS: "{{.*}}save-stats.c"
// NO-STATS-NO: -stats-file

// RUN: %clang -target x86_64-apple-darwin -save-stats=obj -c -o obj/dir/save-stats.o %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-OBJ
// CHECK-OBJ: "-stats-file=obj/dir{{/|\\\\}}save-stats.stats"
// CHECK-OBJ: "-o" "obj/dir{{/|\\\\}}save-stats.o"

// RUN: %clang -target x86_64-apple-darwin -save-stats=obj -c %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-OBJ-NOO
// CHECK-OBJ-NOO: "-stats-file=save-stats.stats"
// CHECK-OBJ-NOO: "-o" "save-stats.o"

// RUN: %clang -target x86_64-apple-darwin -save-stats=bla -c %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID
// CHECK-INVALID: invalid value 'bla' in '-save-stats=bla'
