/* An op-code of 0 is guaranteed to be illegal.  */
#define ABORT_INSTRUCTION asm (".long 0")
