// RUN: touch %t.o

// RUN: %clang -target x86_64-unknown-linux -### %t.o -flto=thin \
// RUN:   -fprofile-use 2>&1 | FileCheck %s

// CHECK: -plugin-opt=cs-profile-path=default.profdata
