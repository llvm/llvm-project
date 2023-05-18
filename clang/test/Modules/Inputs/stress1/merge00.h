#ifndef STRESS1_MERGE00_H
#define STRESS1_MERGE00_H

// These don't match the imported declarations because we import them from
// modules which are built in isolation of the current header's pragma state
// much like they are built in isolation of the incoming macro state.
// FIXME: We should expect warnings here but we can't because verify doesn't
// work for modules.
//#pragma weak pragma_weak01 // expected-warning {{weak identifier 'pragma_weak01' never declared}}
//#pragma weak pragma_weak04 // expected-warning {{weak identifier 'pragma_waek04' never declared}}

#ifdef MERGE_NO_REEXPORT
#include "merge_no_reexport.h"
#endif

#include "common.h"
#include "m00.h"
#include "m01.h"
#include "m02.h"
#include "m03.h"

inline int g() { return N00::S00('a').method00('b') + (int)S00(42) + function00(42); }

// Use implicit special memebers again for S01 to ensure that we merge them in
// successfully from m01.
inline N00::S01 h() { return N00::S01(); }

#pragma weak pragma_weak02
#pragma weak pragma_weak05

extern "C" int pragma_weak02();
int pragma_weak05;

#endif
