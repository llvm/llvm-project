// RUN: %clang -E -MD -MF - %s | FileCheck -check-prefix ZERO_AND_HAS_INCLUDE %s
//
// ZERO_AND_HAS_INCLUDE-NOT: limits.h
//
#if 0 && __has_include(<limits.h>)
#include <limits.h>
#endif

#if 4==5 && __has_include(<limits.h>)
#include <limits.h>
#endif

#if defined(_THIS_IS_NOT_DEFINED) && __has_include(<limits.h>)
#include <limits.h>
#endif

#if 0 && (__has_include(<limits.h>))
#include <limits.h>
#endif

#if 4==5 && (__has_include(<limits.h>))
#include <limits.h>
#endif

#if defined(_THIS_IS_NOT_DEFINED) && (__has_include(<limits.h>))
#include <limits.h>
#endif

#if 0 && (5==5 && __has_include(<limits.h>))
#include <limits.h>
#endif

#if 1 && (4==5 && __has_include(<limits.h>))
#include <limits.h>
#endif






#if 1 || __has_include(<limits.h>)
XXXXXXXXXX
#endif
#if 5==5 || __has_include(<limits.h>)
XXXXXXXXXX
#endif

#if defined(__clang__) || __has_include(<limits.h>)
XXXXXXXXXX
#endif

#if 1 || (__has_include(<limits.h>))
#endif

#if 5==5 || (__has_include(<limits.h>))
XXXXXXXXXX
#endif

#if defined(__clang__) || (__has_include(<limits.h>))
XXXXXXXXXX
#endif

#if 1 && (5==5 || __has_include(<limits.h>))
XXXXXXXXXX
#endif

#if 1 || (5==5 || __has_include(<limits.h>))
XXXXXXXXXX
#endif

#if 0 || (5==5 || __has_include(<limits.h>))
XXXXXXXXXX
#endif
