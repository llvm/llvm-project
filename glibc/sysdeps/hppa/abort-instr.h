/* An instruction privileged instruction to crash a userspace program.

   We go with iitlbp because it has a history of being used to crash
   programs.  */

#define ABORT_INSTRUCTION asm ("iitlbp %r0,(%sr0, %r0)")
