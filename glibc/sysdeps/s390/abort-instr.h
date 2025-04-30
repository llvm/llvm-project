/* An op-code of 0 should crash any program.  */
#define ABORT_INSTRUCTION __asm__ (".word 0")
