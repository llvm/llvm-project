#define cool

#if defined(cool)

#if defined(really_cool)
#endif // really_cool

#elif defined(hot)
// hot


#endif // trailing comment

#ifndef cool
#ifndef uncool

int probably_hot = 1;

#endif // uncool
#endif // cool

// RUN: env CINDEXTEST_SHOW_SKIPPED_RANGES=1 c-index-test -test-annotate-tokens=%s:1:1:16:1 %s | FileCheck %s
// CHECK: Skipping: [5:2 - 6:7]
// CHECK: Skipping: [8:2 - 12:7]
// CHECK: Skipping: [14:2 - 20:7]
