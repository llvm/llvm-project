#include <vector>
#include "generated/gen.inc"

SpelledGen G;

// RUN: clang-include-cleaner -print=changes %s --fragment-headers='generated/gen\.inc' -- -I%S/Inputs/ | count 0
