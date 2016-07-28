; RUN: %clang -target armv7-windows -c -### %s 2>&1 | FileCheck %s
; CHECK: "-triple" "thumbv7-
