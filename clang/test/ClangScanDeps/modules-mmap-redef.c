// Test duplicating module definitions in the same modulemap.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: not clang-scan-deps -compilation-database %t/cdb.json -format \
// RUN:   experimental-full -module-names=A > %t/result.json 2> %t/err.txt
// RUN: cat %t/err.txt | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t \
// RUN:   -check-prefix=ERROR %s

// ERROR: include/A/module.modulemap:9:8: error: redefinition of module 'A'
// ERROR-NEXT: include/A/module.modulemap:1:8: note: previously defined here

//--- include/A/module.modulemap
module A {
    header "A.h"
}

module A1 {
    header "A1.h"
}

module A{
    header "A.h"
}

//--- include/A/A.h

//--- include/A/A1.h

//--- include/A/A2.h


//--- cdb.json.template
[{
  "file": "",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -I DIR/include/A -x c"
}]
