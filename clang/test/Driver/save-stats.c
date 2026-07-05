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

// RUN: not %clang -target x86_64-apple-darwin -save-stats=bla -c %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID
// CHECK-INVALID: invalid value 'bla' in '-save-stats=bla'

// RUN: %clang --target=x86_64-unknown-linux -save-stats -flto -o obj/dir/save-stats.exe %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-LTO
// Previously `-plugin-opt=stats-file` would use empty filename if a linker flag (i.e. -Wl) is presented before any input filename.
// RUN: %clang --target=x86_64-unknown-linux -save-stats -flto -o obj/dir/save-stats.exe -Wl,-plugin-opt=-dummy %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-LTO
// RUN: %clang --target=x86_64-unknown-freebsd -save-stats -flto -o obj/dir/save-stats.exe %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-LTO
// RUN: %clang --target=x86_64-unknown-freebsd -save-stats -flto -o obj/dir/save-stats.exe -Wl,-plugin-opt=-dummy %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-LTO
// RUN: %clang --target=x86_64-unknown-openbsd -save-stats -flto -o obj/dir/save-stats.exe %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-LTO
// RUN: %clang --target=x86_64-unknown-openbsd -save-stats -flto -o obj/dir/save-stats.exe -Wl,-plugin-opt=-dummy %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-LTO
// RUN: %clang --target=x86_64-unknown-fuchsia -save-stats -flto -o obj/dir/save-stats.exe %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-LTO
// RUN: %clang --target=x86_64-unknown-fuchsia -save-stats -flto -o obj/dir/save-stats.exe -Wl,-plugin-opt=-dummy %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-LTO
// RUN: %clang --target=x86_64-unknown-haiku -save-stats -flto -o obj/dir/save-stats.exe %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-LTO
// RUN: %clang --target=x86_64-unknown-haiku -save-stats -flto -o obj/dir/save-stats.exe -Wl,-plugin-opt=-dummy %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-LTO
// RUN: %clang --target=avr -save-stats -flto -o obj/dir/save-stats.exe %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-LTO
// RUN: %clang --target=avr -save-stats -flto -o obj/dir/save-stats.exe -Wl,-plugin-opt=-dummy %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-LTO
// CHECK-LTO: "-stats-file=save-stats.stats"
// CHECK-LTO: "-o" "obj/dir{{/|\\\\}}save-stats.exe"
// CHECK-LTO: "-plugin-opt=stats-file=save-stats.stats"

// RUN: %clang --target=powerpc-unknown-aix  -save-stats -flto -o obj/dir/save-stats %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-AIX-LTO
// RUN: %clang --target=powerpc-unknown-aix  -save-stats -flto -o obj/dir/save-stats -Wl,-bplugin-opt:-dummy %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-AIX-LTO

// CHECK-AIX-LTO: "-stats-file=save-stats.stats"
// CHECK-AIX-LTO: "-o" "obj/dir{{/|\\\\}}save-stats"
// CHECK-AIX-LTO: "-bplugin_opt:stats-file=save-stats.stats"

// RUN: %clang --target=x86_64-unknown-linux -save-stats=obj -flto -o obj/dir/save-stats.exe %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-LTO-OBJ
// CHECK-LTO-OBJ: "-plugin-opt=stats-file=obj/dir{{/|\\\\}}save-stats.stats"

// RUN: env CC_PRINT_INTERNAL_STAT=1 \
// RUN: %clang -target x86_64-apple-darwin %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-ENV
// CHECK-ENV: "-stats-file=-"
// CHECK-ENV-NO: "stats-file-append"

// RUN: env CC_PRINT_INTERNAL_STAT=1 \
// RUN:     CC_PRINT_INTERNAL_STAT_FILE=/tmp/stats.json \
// RUN: %clang -target x86_64-apple-darwin %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-ENV-FILE
// CHECK-ENV-FILE: "-stats-file=/tmp/stats.json"
// CHECK-ENV-FILE: "-stats-file-append"
