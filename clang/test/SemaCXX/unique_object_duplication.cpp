// RUN: %clang_cc1 -fsyntax-only -Wunique-object-duplication -Wno-unused-value \
// RUN:   -verify -triple=x86_64-pc-linux-gnu %s
// RUN: %clang_cc1 -fsyntax-only -Wunique-object-duplication -Wno-unused-value \
// RUN:   -verify=hidden -triple=x86_64-pc-linux-gnu -fvisibility=hidden  %s
// RUN: %clang_cc1 -fsyntax-only -Wunique-object-duplication -Wno-unused-value \
// RUN:   -verify=windows -triple=x86_64-windows-msvc -DWINDOWS_TEST -fdeclspec %s

#include "unique_object_duplication.h"

// Everything in these namespaces here is defined in the cpp file,
// so won't get duplicated

namespace GlobalTest {
  float Test::allowedStaticMember1 = 2.3;
}

bool disallowed4 = true;
constexpr inline bool disallowed5 = true;