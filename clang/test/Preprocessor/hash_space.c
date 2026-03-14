// RUN: %clang_cc1 %s -E | grep " #"

// Should put a space before the # so that -fpreprocessed mode doesn't
// macro expand this again.
#define HASH #
HASH define foo bar
