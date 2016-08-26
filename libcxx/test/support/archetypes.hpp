#ifndef TEST_SUPPORT_ARCHETYPES_HPP
#define TEST_SUPPORT_ARCHETYPES_HPP

#include "test_macros.h"

#if TEST_STD_VER >= 11

//============================================================================//
// Trivial Implicit Test Types
namespace ImplicitTypes {
#include "archetypes.ipp"
}

//============================================================================//
// Trivial Explicit Test Types
namespace ExplicitTypes {
#define DEFINE_EXPLICIT explicit
#include "archetypes.ipp"
}

//============================================================================//
// Non-Trivial Implicit Test Types
namespace NonLiteralTypes {
#define DEFINE_DTOR(Name) ~Name() {}
#include "archetypes.ipp"
}

//============================================================================//
// Non-Trivially Copyable Implicit Test Types
namespace NonTrivialTypes {
#define DEFINE_CTOR {}
#include "archetypes.ipp"
}

#endif // TEST_STD_VER >= 11

#endif // TEST_SUPPORT_ARCHETYPES_HPP
