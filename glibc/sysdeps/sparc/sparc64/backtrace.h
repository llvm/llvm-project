/* Private macros for guiding the backtrace implementation, sparc64
   version.  */

#define backtrace_flush_register_windows() \
	asm volatile ("flushw")

#define BACKTRACE_STACK_BIAS	STACK_BIAS
