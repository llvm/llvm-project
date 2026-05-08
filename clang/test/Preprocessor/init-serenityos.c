// RUN: %clang_cc1 -triple x86_64-unknown-serenity -E -dM < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix SERENITY %s
// SERENITY: #define __serenity__ 1
// SERENITY: #define __unix__ 1
