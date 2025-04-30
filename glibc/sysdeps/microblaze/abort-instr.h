/* An instruction which should crash any program is `illegal'.  */
#define ABORT_INSTRUCTION asm ("brki r0, -1")
