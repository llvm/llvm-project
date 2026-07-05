#include "a2.h"

// Add update record for definition of A<int> and constructors.
// We need an eagerly-emitted use here to get the problematic
// deserialization ordering.
N::A<int> b2;
