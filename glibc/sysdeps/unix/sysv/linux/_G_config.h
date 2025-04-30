/* Configuration parameters for stdio - Linux version.  */

#ifndef __G_CONFIG_H
#define __G_CONFIG_H 1

/* Define to 1 if the operating system supports mmap, 0 otherwise.
   This function is required by POSIX but might still be unavailable,
   for instance when the hardware lacks support for virtual memory.  */
#define _G_HAVE_MMAP 1

/* Define to 1 if the operating system supports mremap, 0 otherwise.
   This function is currently a Linux-specific extension.  */
#define _G_HAVE_MREMAP 1

#endif	/* bits/_G_config.h */
