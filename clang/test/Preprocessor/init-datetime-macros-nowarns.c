// RUN: %clang_cc1 -E -DDATETIME_CUSTOM -init-datetime-macros=undefined -D__DATE__="\"d3\"" -D__TIME__="\"t4\"" -D__TIMESTAMP__="\"ts5\"" -verify %s
// expected-no-diagnostics

date: __DATE__
time: __TIME__
timestamp: __TIMESTAMP__
