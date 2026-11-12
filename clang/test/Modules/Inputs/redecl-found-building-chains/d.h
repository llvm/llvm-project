#include "a.h" // ensure that our canonical decl is not from b
struct A;
#include "b.h"
struct A;
#include "c.h" // ensure that our type for A doesn't reference the definition in b
struct A;
