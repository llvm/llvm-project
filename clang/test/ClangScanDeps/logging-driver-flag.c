// UNSUPPORTED: system-windows
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: sed -e "s|DIR|%/t|g" %t/cdb-by-name.json.template > %t/cdb-by-name.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full -j 1 -o %t/deps.json
// RUN: FileCheck %s --check-prefix=TU --input-file %t/tu.log

// TU: [{{[0-9]+\.[0-9]+}}] [[#PID:]] [[#TID:]]: starting scanning command:{{.*}}tu.c
// TU: [{{[0-9]+\.[0-9]+}}] {{.*}}: pcm_write: {{.*}}.pcm
// TU: [{{[0-9]+\.[0-9]+}}] {{.*}}: finished scanning command:{{.*}}tu.c

// RUN: clang-scan-deps -compilation-database %t/cdb-by-name.json \
// RUN:   -format experimental-full -j 1 -module-names=A
// RUN: FileCheck %s --check-prefix=BY-NAME --input-file %t/by-name.log

// BY-NAME: [{{[0-9]+\.[0-9]+}}] {{.*}}: start scan_by_name: A
// BY-NAME: [{{[0-9]+\.[0-9]+}}] {{.*}}: finish scan_by_name: A

//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -fbuild-session-timestamp=1
-fmodules-validate-once-per-build-session -fdepscan-log-path=DIR/tu.log",
  "file": "DIR/tu.c"
}]

//--- cdb-by-name.json.template
[{
  "directory": "DIR",
  "command": "clang -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -I DIR -x c -fdepscan-log-path=DIR/by-name.log",
  "file": ""
}]

//--- module.modulemap
module A { header "A.h" }
//--- A.h
void A_func(void);
//--- tu.c
#include "A.h"
void foo(void) { A_func(); }
