// RUN: %clang_cc1 -E %s | grep bark

#if 0
blah
#\
else
bark
#endif

