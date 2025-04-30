/* Private macros for guiding the backtrace implementation, sparc32
   version.  */

#define backtrace_flush_register_windows() \
	asm volatile ("ta %0" : : "i" (ST_FLUSH_WINDOWS))

#define BACKTRACE_STACK_BIAS	0
