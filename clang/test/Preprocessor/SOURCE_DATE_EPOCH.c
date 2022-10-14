// RUN: env SOURCE_DATE_EPOCH=0 %clang_cc1 -E %s | FileCheck %s --check-prefix=19700101

// 19700101:      const char date[] = "Jan  1 1970";
// 19700101-NEXT: const char time[] = "00:00:00";
// 19700101-NEXT: const char timestamp[] = "Thu Jan  1 00:00:00 1970";

// RUN: env SOURCE_DATE_EPOCH=2147483647 %clang_cc1 -E -Wdate-time %s 2>&1 | FileCheck %s --check-prefix=Y2038

// Y2038:      warning: expansion of date or time macro is not reproducible [-Wdate-time]
// Y2038:      const char date[] = "Jan 19 2038";
// Y2038-NEXT: const char time[] = "03:14:07";
// Y2038-NEXT: const char timestamp[] = "Tue Jan 19 03:14:07 2038";

/// Test a large timestamp if the system uses 64-bit time_t and known to support large timestamps.
// RUN: %if !system-windows && clang-target-64-bits %{ env SOURCE_DATE_EPOCH=253402300799 %clang_cc1 -E -Wdate-time %s 2>&1 | FileCheck %s --check-prefix=99991231 %}

// 99991231:      warning: expansion of date or time macro is not reproducible [-Wdate-time]
// 99991231:      const char date[] = "Dec 31 9999";
// 99991231-NEXT: const char time[] = "23:59:59";
// 99991231-NEXT: const char timestamp[] = "Fri Dec 31 23:59:59 9999";

// RUN: env SOURCE_DATE_EPOCH=253402300800 not %clang_cc1 -E %s 2>&1 | FileCheck %s --check-prefix=TOOBIG

// TOOBIG: error: environment variable 'SOURCE_DATE_EPOCH' ('253402300800') must be a non-negative decimal integer <= {{(2147483647|253402300799)}}

// RUN: env SOURCE_DATE_EPOCH=0x0 not %clang_cc1 -E %s 2>&1 | FileCheck %s --check-prefix=NOTDECIMAL

// NOTDECIMAL: error: environment variable 'SOURCE_DATE_EPOCH' ('0x0') must be a non-negative decimal integer <= {{(2147483647|253402300799)}}

const char date[] = __DATE__;
const char time[] = __TIME__;
const char timestamp[] = __TIMESTAMP__;
