typedef union {} pthread_mutex_t;

// Define then merge with another definition.
typedef struct {} merged_after_definition;
#include "c1.h"
