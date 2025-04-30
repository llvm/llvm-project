/* An instruction which should crash any program is `sleep'.  */
#define ABORT_INSTRUCTION_ASM sleep
#define ABORT_INSTRUCTION asm ("sleep")
