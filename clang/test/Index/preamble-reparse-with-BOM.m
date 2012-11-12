
@interface I2
@end

// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_FAILONERROR=1 \
// RUN:   c-index-test -test-load-source-reparse 1 local %s
