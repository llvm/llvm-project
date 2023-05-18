#define MODULE
#define INTEGER(X) int
#define FLOAT float
#define DOUBLE double

#__public_macro INTEGER
#__private_macro FLOAT
#__private_macro MODULE

int (INTEGER);

#if !__building_module(macros)
#  error Can't include this header without building the 'macros' module.
#endif

#ifdef __MODULE__
extern int __MODULE__;
#endif

#include "macros-indirect.h"
