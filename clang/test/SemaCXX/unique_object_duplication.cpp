// RUN: %clang_cc1 -fsyntax-only -verify=hidden -Wunique-object-duplication -fvisibility=hidden -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wunique-object-duplication -Wno-unused-value %s
// The check is currently disabled on windows. The test should fail because we're not getting the expected warnings.
// XFAIL: target={{.*}}-windows{{.*}}, {{.*}}-ps{{(4|5)(-.+)?}}

#include "unique_object_duplication.h"

// Everything in these namespaces here is defined in the cpp file,
// so won't get duplicated

namespace GlobalTest {
  float Test::allowedStaticMember1 = 2.3;
}

bool disallowed4 = true;
constexpr inline bool disallowed5 = true;