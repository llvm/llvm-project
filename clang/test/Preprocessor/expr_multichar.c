// RUN: %clang_cc1 < %s -E -verify -triple i686-pc-linux-gnu
// expected-no-diagnostics

#if (('1234' >> 24) != '1')
#error Bad multichar constant calculation!
#endif
