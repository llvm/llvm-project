/* An instruction which should crash any program is a breakpoint.  */
#ifdef __mips16
# define ABORT_INSTRUCTION asm ("break 63")
#else
# define ABORT_INSTRUCTION asm ("break 255")
#endif
