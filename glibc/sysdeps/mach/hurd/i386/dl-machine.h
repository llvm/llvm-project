/* Dynamic linker magic for Hurd/i386.
   This file just gets us a call to _dl_first_init inserted
   into the asm in sysdeps/i386/dl-machine.h that contains
   the initializer code.  */

#define RTLD_START_SPECIAL_INIT "call _dl_init_first@PLT; movl (%esp), %edx"
#include_next "dl-machine.h"
