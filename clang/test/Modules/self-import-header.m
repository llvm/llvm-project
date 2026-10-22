// REQUIRES: x86-registered-target
// RUN: rm -rf %t
// RUN: %clang -fsyntax-only -isysroot %S/Inputs/System/usr/include -fmodules -fmodules-cache-path=%t \
// RUN:    -target x86_64-darwin \
// RUN:    -F %S -F %S/Inputs/self-import-header -I %S/Inputs/self-import-header/depend_builtin \
// RUN:    %s -D__need_wint_t -Werror=implicit-function-declaration

@import af;
