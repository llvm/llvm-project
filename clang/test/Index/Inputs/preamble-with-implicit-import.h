#include "preamble-with-implicit-import-A.h"

// Typo is defined in B, which is not imported.
void useTypeFromB(Typo *);
