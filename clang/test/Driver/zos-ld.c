// General tests that ld invocations for z/OS are valid.

// 1. General C link for executable
// RUN: %clang -### --target=s390x-ibm-zos %s 2>&1 \
// RUN:   | FileCheck --check-prefix=C-LD %s

// C-LD: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// C-LD: "AMODE=64,LIST,DYNAM=DLL,MSGLEVEL=4,CASE=MIXED,REUS=RENT"
// C-LD-SAME: "-e" "CELQSTRT"
// C-LD-SAME: "-O" "CELQSTRT"
// C-LD-SAME: "-u" "CELQMAIN"
// C-LD-SAME: "-x" "/dev/null"
// C-LD-SAME: "-S" "//'CEE.SCEEBND2'"
// C-LD-SAME: "-S" "//'SYS1.CSSLIB'"
// C-LD-SAME: "//'CEE.SCEELIB(CELQS001)'"
// C-LD-SAME: "//'CEE.SCEELIB(CELQS003)'"
// C-LD-SAME: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}zos{{/|\\\\}}libclang_rt.builtins-s390x.a"

// 2. General C link for dll
// RUN: %clang -### --shared --target=s390x-ibm-zos %s 2>&1 \
// RUN:   | FileCheck --check-prefix=C-LD-DLL %s

// C-LD-DLL: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// C-LD-DLL: "AMODE=64,LIST,DYNAM=DLL,MSGLEVEL=4,CASE=MIXED,REUS=RENT"
// C-LD-DLL-NOT: "-e" "CELQSTRT"
// C-LD-DLL-NOT: "-O" "CELQSTRT"
// C-LD-DLL-NOT: "-u" "CELQMAIN"
// C-LD-DLL-SAME: "-x" "{{.*}}.x"
// C-LD-DLL-SAME: "-S" "//'CEE.SCEEBND2'"
// C-LD-DLL-SAME: "-S" "//'SYS1.CSSLIB'"
// C-LD-DLL-SAME: "//'CEE.SCEELIB(CELQS001)'"
// C-LD-DLL-SAME: "//'CEE.SCEELIB(CELQS003)'"
// C-LD-DLL-SAME: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}zos{{/|\\\\}}libclang_rt.builtins-s390x.a"

// 3. General C++ link for executable
// RUN: %clangxx -### --target=s390x-ibm-zos %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CXX-LD %s

// CXX-LD: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CXX-LD: "AMODE=64,LIST,DYNAM=DLL,MSGLEVEL=4,CASE=MIXED,REUS=RENT"
// CXX-LD-SAME: "-e" "CELQSTRT"
// CXX-LD-SAME: "-O" "CELQSTRT"
// CXX-LD-SAME: "-u" "CELQMAIN"
// CXX-LD-SAME: "-x" "/dev/null"
// CXX-LD-SAME: "-S" "//'CEE.SCEEBND2'"
// CXX-LD-SAME: "-S" "//'SYS1.CSSLIB'"
// CXX-LD-SAME: "//'CEE.SCEELIB(CELQS001)'"
// CXX-LD-SAME: "//'CEE.SCEELIB(CELQS003)'"
// CXX-LD-SAME: "//'CEE.SCEELIB(CRTDQCXE)'"
// CXX-LD-SAME: "//'CEE.SCEELIB(CRTDQCXS)'"
// CXX-LD-SAME: "//'CEE.SCEELIB(CRTDQCXP)'"
// CXX-LD-SAME: "//'CEE.SCEELIB(CRTDQCXA)'"
// CXX-LD-SAME: "//'CEE.SCEELIB(CRTDQXLA)'"
// CXX-LD-SAME: "//'CEE.SCEELIB(CRTDQUNW)'"
// CXX-LD-SAME: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}zos{{/|\\\\}}libclang_rt.builtins-s390x.a"

// 4. General C++ link for dll
// RUN: %clangxx -### --shared --target=s390x-ibm-zos %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CXX-LD-DLL %s

// CXX-LD-DLL: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CXX-LD-DLL: "AMODE=64,LIST,DYNAM=DLL,MSGLEVEL=4,CASE=MIXED,REUS=RENT"
// CXX-LD-DLL-NOT: "-e" "CELQSTRT"
// CXX-LD-DLL-NOT: "-O" "CELQSTRT"
// CXX-LD-DLL-NOT: "-u" "CELQMAIN"
// CXX-LD-DLL-SAME: "-x" "{{.*}}.x"
// CXX-LD-DLL-SAME: "-S" "//'CEE.SCEEBND2'"
// CXX-LD-DLL-SAME: "-S" "//'SYS1.CSSLIB'"
// CXX-LD-DLL-SAME: "//'CEE.SCEELIB(CELQS001)'"
// CXX-LD-DLL-SAME: "//'CEE.SCEELIB(CELQS003)'"
// CXX-LD-DLL-SAME: "//'CEE.SCEELIB(CRTDQCXE)'"
// CXX-LD-DLL-SAME: "//'CEE.SCEELIB(CRTDQCXS)'"
// CXX-LD-DLL-SAME: "//'CEE.SCEELIB(CRTDQCXP)'"
// CXX-LD-DLL-SAME: "//'CEE.SCEELIB(CRTDQCXA)'"
// CXX-LD-DLL-SAME: "//'CEE.SCEELIB(CRTDQXLA)'"
// CXX-LD-DLL-SAME: "//'CEE.SCEELIB(CRTDQUNW)'"
// CXX-LD-DLL-SAME: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}zos{{/|\\\\}}libclang_rt.builtins-s390x.a"

// 5. C++ link for executable w/ -mzos-hlq-le=, -mzos-hlq-csslib=
// RUN: %clangxx -### --target=s390x-ibm-zos %s 2>&1 \
// RUN:   -mzos-hlq-le=AAAA -mzos-hlq-csslib=BBBB \
// RUN:   | FileCheck --check-prefix=CXX-LD5 %s

// CXX-LD5: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CXX-LD5: "AMODE=64,LIST,DYNAM=DLL,MSGLEVEL=4,CASE=MIXED,REUS=RENT"
// CXX-LD5-SAME: "-e" "CELQSTRT"
// CXX-LD5-SAME: "-O" "CELQSTRT"
// CXX-LD5-SAME: "-u" "CELQMAIN"
// CXX-LD5-SAME: "-x" "/dev/null"
// CXX-LD5-SAME: "-S" "//'AAAA.SCEEBND2'"
// CXX-LD5-SAME: "-S" "//'BBBB.CSSLIB'"
// CXX-LD5-SAME: "//'AAAA.SCEELIB(CELQS001)'"
// CXX-LD5-SAME: "//'AAAA.SCEELIB(CELQS003)'"
// CXX-LD5-SAME: "//'AAAA.SCEELIB(CRTDQCXE)'"
// CXX-LD5-SAME: "//'AAAA.SCEELIB(CRTDQCXS)'"
// CXX-LD5-SAME: "//'AAAA.SCEELIB(CRTDQCXP)'"
// CXX-LD5-SAME: "//'AAAA.SCEELIB(CRTDQCXA)'"
// CXX-LD5-SAME: "//'AAAA.SCEELIB(CRTDQXLA)'"
// CXX-LD5-SAME: "//'AAAA.SCEELIB(CRTDQUNW)'"
// CXX-LD5-SAME: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}zos{{/|\\\\}}libclang_rt.builtins-s390x.a"

// 6. C++ link for executable w/ -mzos-hlq-clang=
// RUN: %clangxx -### --target=s390x-ibm-zos %s 2>&1 \
// RUN:   -mzos-hlq-clang=AAAA \
// RUN:   | FileCheck --check-prefix=CXX-LD6 %s

// CXX-LD6: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CXX-LD6: "AMODE=64,LIST,DYNAM=DLL,MSGLEVEL=4,CASE=MIXED,REUS=RENT"
// CXX-LD6-SAME: "-e" "CELQSTRT"
// CXX-LD6-SAME: "-O" "CELQSTRT"
// CXX-LD6-SAME: "-u" "CELQMAIN"
// CXX-LD6-SAME: "-x" "/dev/null"
// CXX-LD6-SAME: "-S" "//'CEE.SCEEBND2'"
// CXX-LD6-SAME: "-S" "//'SYS1.CSSLIB'"
// CXX-LD6-SAME: "//'CEE.SCEELIB(CELQS001)'"
// CXX-LD6-SAME: "//'CEE.SCEELIB(CELQS003)'"
// CXX-LD6-SAME: "//'AAAA.SCEELIB(CRTDQCXE)'"
// CXX-LD6-SAME: "//'AAAA.SCEELIB(CRTDQCXS)'"
// CXX-LD6-SAME: "//'AAAA.SCEELIB(CRTDQCXP)'"
// CXX-LD6-SAME: "//'AAAA.SCEELIB(CRTDQCXA)'"
// CXX-LD6-SAME: "//'AAAA.SCEELIB(CRTDQXLA)'"
// CXX-LD6-SAME: "//'AAAA.SCEELIB(CRTDQUNW)'"
// CXX-LD6-SAME: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}zos{{/|\\\\}}libclang_rt.builtins-s390x.a"
