#include <vector>
#include "gen.inc"

Gen G;

// RUN: clang-include-cleaner -print=changes %s --fragment-headers='.*\.inc$' -- -I%S/Inputs/ | count 0
