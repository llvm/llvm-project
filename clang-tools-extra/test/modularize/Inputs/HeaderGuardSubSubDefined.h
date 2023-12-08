#if !defined(_HEADERGUARDSUBSUBDEFINED_H_)
#define _HEADERGUARDSUBSUBDEFINED_H_

#define SOMETHING_OTHER 1

// Nest include.  Header guard should not confuse modularize.
#include "HeaderGuard.h"

#endif // _HEADERGUARDSUBSUBDEFINED_H_
