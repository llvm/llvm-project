// RUN: rm -f %t.hmap
// RUN: %hmaptool write %S/Inputs/include-header-missing-in-framework/TestFramework.hmap.json %t.hmap
// RUN: %clang_cc1 -fsyntax-only -F %S/Inputs -I %t.hmap -verify %s -DLATE_REMAPPING
// RUN: %clang_cc1 -fsyntax-only -I %t.hmap -F %S/Inputs -verify %s

// The test is similar to 'include-header-missing-in-framework.c' but covers
// the case when a header is remapped to a framework-like path with a .hmap
// file. And we can find the framework but not the header.

#ifdef LATE_REMAPPING
// Framework is found before remapping.
#include <TestFramework/BeforeRemapping.h>
// expected-error@-1 {{'TestFramework/BeforeRemapping.h' file not found}}
// expected-note@-2 {{did not find header 'BeforeRemapping.h' in framework 'TestFramework' (loaded from}}

#else
// Framework is found after remapping.
#include "RemappedHeader.h"
// expected-error@-1 {{'RemappedHeader.h' file not found}}
#endif // LATE_REMAPPING
