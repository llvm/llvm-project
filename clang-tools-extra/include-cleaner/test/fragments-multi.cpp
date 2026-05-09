#include <vector>
#include "a.inc"
#include "b.inc"

A AValue;
B BValue;

// RUN: clang-include-cleaner -print=changes %s --fragment-headers='.*\.inc$' -- -I%S/Inputs/ | count 0
