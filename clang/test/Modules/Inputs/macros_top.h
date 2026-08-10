#define TOP unsigned int

#define TOP_LEFT_UNDEF 1







#define TOP_RIGHT_REDEF float
// The last definition will be exported from the sub-module.
#define TOP_RIGHT_REDEF int

#define TOP_RIGHT_UNDEF int

#define TOP_OTHER_UNDEF1 42
#undef TOP_OTHER_UNDEF2
#define TOP_OTHER_REDEF1 1
#define TOP_OTHER_REDEF2 2

#define TOP_OTHER_DEF_RIGHT_UNDEF void

#define TOP_REDEF_IN_SUBMODULES 0
#define TOP_DEF_RIGHT_UNDEF void
