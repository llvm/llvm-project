// Test -MF and -E flags with has_include

#ifdef TEST_HAS_INCLUDE_NEXT
#if __has_include_next(<limits.h>)
// DO NOTHING
#endif
#endif

#ifdef TEST_HAS_INCLUDE
#if __has_include(<limits.h>)
// DO NOTHING
#endif
#endif

// RUN: %clang -DTEST_HAS_INCLUDE -E -MD -MF - %s \
// RUN:    | FileCheck -check-prefix=TEST-HAS %s
// TEST-HAS: dependencies-on-has-include.o:
// TEST-HAS-NOT: limits.h

// RUN: %clang -Wno-include-next-outside-header -DTEST_HAS_INCLUDE_NEXT -E -MD -MF - %s \
// RUN:    | FileCheck -check-prefix=TEST-HAS-N %s
// TEST-HAS-N: dependencies-on-has-include.o:
// TEST-HAS-N-NOT: limits.h
