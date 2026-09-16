// RUN: %clang_cc1 -E -DDATETIME_CUSTOM -init-datetime-macros=literalone -D__DATE__="\"d3\"" -D__TIME__="\"t4\"" -D__TIMESTAMP__="\"ts5\"" -verify %s
// expected-warning@2{{redefining builtin macro}}
// expected-warning@3{{redefining builtin macro}}
// expected-warning@4{{redefining builtin macro}}

date: __DATE__
time: __TIME__
timestamp: __TIMESTAMP__
