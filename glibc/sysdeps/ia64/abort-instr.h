/* An instruction which should crash any program is `break 0' which triggers
   SIGILL.  */
#define ABORT_INSTRUCTION asm ("break 0")
