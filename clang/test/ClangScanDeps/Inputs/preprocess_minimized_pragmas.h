// #pragma push_macro/pop_macro
#define INCLUDE_A
#pragma push_macro("INCLUDE_A")
#undef INCLUDE_A
#pragma pop_macro("INCLUDE_A")

#ifdef INCLUDE_A
#include "a.h"
#endif

// #pragma push_macro/pop_macro with argument macro expansion
#define INCLUDE_B
#define MACRO_NAME "INCLUDE_B"

#pragma push_macro(MACRO_NAME)
#undef INCLUDE_B
#pragma pop_macro(MACRO_NAME)

#ifdef INCLUDE_B
#include "b.h"
#endif

// #pragma include_alias (MS specific)
// When compiling without MS Extensions, the pragma is not recognized,
// and the file c_alias.h is included instead of c.h
#pragma include_alias("c_alias.h", "c.h")
#include "c_alias.h"
