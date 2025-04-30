/* An instruction which should crash any program is an unused trap.  */
#define ABORT_INSTRUCTION asm ("trap 31")
