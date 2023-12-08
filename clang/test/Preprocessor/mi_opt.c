// RUN: not %clang_cc1 -fsyntax-only %s
// PR1900
// This test should get a redefinition error from m_iopt.h: the MI opt 
// shouldn't apply.

#define MACRO
#include "mi_opt.h"
#undef MACRO
#define MACRO || 1
#include "mi_opt.h"

