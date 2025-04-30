/* FLAG 1 is privilege mode only instruction, hence will crash any program.  */
#define ABORT_INSTRUCTION asm ("flag 1")
