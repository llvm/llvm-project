struct X { X(); virtual ~X(); };
inline X::~X() {}
#include "a.h"
