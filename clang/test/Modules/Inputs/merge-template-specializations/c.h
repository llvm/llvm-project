#include "a.h"
struct X { SmallString<256> ss; };

template<typename T> void use_var_template(decltype(T() + var_template<0>));

#include "b.h"
