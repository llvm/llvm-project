/* An instruction which should crash any program is an unimp.  */
#define ABORT_INSTRUCTION asm ("unimp 0xf00")
