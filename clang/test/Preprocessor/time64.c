// RUN: %clang_cc1 -E -dM -triple=i686-pc-linux-gnu /dev/null | FileCheck -match-full-lines -check-prefix TIME32 %s
// RUN: %clang_cc1 -E -dM -triple=i686-pc-linux-gnut64 /dev/null | FileCheck -match-full-lines -check-prefix TIME64 %s
// RUN: %clang_cc1 -E -dM -triple=armv5tel-softfloat-linux-gnueabi /dev/null | FileCheck -match-full-lines -check-prefix TIME32 %s
// RUN: %clang_cc1 -E -dM -triple=armv5tel-softfloat-linux-gnueabit64 /dev/null | FileCheck -match-full-lines -check-prefix TIME64 %s
// RUN: %clang_cc1 -E -dM -triple=armv7a-unknown-linux-gnueabihf /dev/null | FileCheck -match-full-lines -check-prefix TIME32 %s
// RUN: %clang_cc1 -E -dM -triple=armv7a-unknown-linux-gnueabihft64 /dev/null | FileCheck -match-full-lines -check-prefix TIME64 %s
//
// TIME32-NOT:#define _FILE_OFFSET_BITS 64
// TIME32-NOT:#define _TIME_BITS 64
//
// TIME64:#define _FILE_OFFSET_BITS 64
// TIME64:#define _TIME_BITS 64
