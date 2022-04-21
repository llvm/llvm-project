#include "a.h"
void f(SmallString<256>&);

template<typename T> void use_var_template(decltype(T() + var_template<0>)) {}
