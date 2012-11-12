#include <Module/Module.h> //expected-warning{{treating #include as an import of module 'Module'}}

#define DEPENDS_ON_MODULE 1
#__private_macro DEPENDS_ON_MODULE

