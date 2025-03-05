# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: echo 'foo { *; }; bar { *; };' > %t.ver
# RUN: ld.lld --version-script %t.ver %t.o -shared -o %t.so 2>&1 | \
# RUN:   FileCheck --check-prefix=MULTVER %s

# RUN: echo '{ global: *; local: *;};' > %t.ver
# RUN: ld.lld --version-script %t.ver %t.o -shared -o %t.so 2>&1 | \
# RUN:   FileCheck --check-prefix=LOCGLOB %s

# RUN: echo 'V1 { global: *; }; V2 { local: *;};' > %t.ver
# RUN: ld.lld --version-script %t.ver %t.o -shared -o %t.so 2>&1 | \
# RUN:   FileCheck --check-prefix=LOCGLOB %s

# RUN: echo 'V1 { local: *; }; V2 { global: *;};' > %t.ver
# RUN: ld.lld --version-script %t.ver %t.o -shared -o %t.so 2>&1 | \
# RUN:   FileCheck --check-prefix=LOCGLOB %s

# RUN: echo 'V1 { local: *; }; V2 { local: *;};' > %t.ver
# RUN: ld.lld --version-script %t.ver %t.o -shared -o %t.so --fatal-warnings

## --retain-symbols-file uses the same internal infrastructure as the support
## for version scripts. Do not show the warings if they both are used.
# RUN: echo 'foo' > %t_retain.txt
# RUN: echo '{ local: *; };' > %t_local.ver
# RUN: echo '{ global: *; };' > %t_global.ver
# RUN: ld.lld --retain-symbols-file=%t_retain.txt --version-script %t_local.ver %t.o -shared -o %t.so --fatal-warnings
# RUN: ld.lld --retain-symbols-file=%t_retain.txt --version-script %t_global.ver %t.o -shared -o %t.so --fatal-warnings

# MULTVER: warning: wildcard pattern '*' is used for multiple version definitions in version script
# LOCGLOB: warning: wildcard pattern '*' is used for both 'local' and 'global' scopes in version script

.globl foo
foo:
