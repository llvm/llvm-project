
#ifndef NO_RECORD_1
struct ReceivesPragma { };
#endif

#ifdef SET_FIRST_HEADER
#pragma pack (16)
#ifndef SET_SECOND_HEADER
// expected-note@-2 2 {{previous '#pragma pack' directive that modifies alignment is here}}
#else
// expected-note@-4 1 {{previous '#pragma pack' directive that modifies alignment is here}}
#endif
// expected-warning@+3 {{non-default #pragma pack value changes the alignment of struct or union members in the included file}}
#endif

#include "pragma-pack2.h"

#ifdef SET_SECOND_HEADER
// expected-warning@-3 {{the current #pragma pack alignment value is modified in the included file}}
#endif

#ifdef PUSH_POP_FIRST_HEADER
// This is fine, we don't change the current value.
#pragma pack (push, 4)

#pragma pack (pop)
#endif
