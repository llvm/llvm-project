#ifndef STRESS1_M01_H
#define STRESS1_M01_H

#include "common.h"

// Trigger the use of special members for a class this is also defined in other
// modules.
inline N00::S01 m01_special_members() { return N00::S01(); }

#endif
