# // null directive and comments before include guard

#ifndef MULTIPLE_INCLUSION_OPT

int foo();

// The position of the define should not matter
#define MULTIPLE_INCLUSION_OPT

int bar();

#endif

#
#
/* Two null directives
   and a multiline comment
   after the #endif */
