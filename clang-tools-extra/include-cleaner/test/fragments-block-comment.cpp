#include <vector> /* keep me */
#include "gen.inc"

Gen G;

// RUN: clang-include-cleaner -print=changes %s --fragment-headers=.*\.inc$ --fragment-dependency-comment-format='needed by {0}' -- -I%S/Inputs/ | count 0
