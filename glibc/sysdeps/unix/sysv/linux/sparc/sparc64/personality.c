/* Work around sign extension bug in the kernel.  */
#define PERSONALITY_TRUNCATE_ARGUMENT
#include <sysdeps/unix/sysv/linux/personality.c>
