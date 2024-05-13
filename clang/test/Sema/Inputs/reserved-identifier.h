int __i_come_from_a_system_header; // no-warning
#define __I_AM_A_SYSTEM_MACRO()    // no-warning

#define SOME_SYSTEM_MACRO() int __i_come_from_a_system_macro
