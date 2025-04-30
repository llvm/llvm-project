#if NOW == SIGILL
P (ILL_ILLOPC, N_("Illegal opcode"))
P (ILL_ILLOPN, N_("Illegal operand"))
P (ILL_ILLADR, N_("Illegal addressing mode"))
P (ILL_ILLTRP, N_("Illegal trap"))
P (ILL_PRVOPC, N_("Privileged opcode"))
P (ILL_PRVREG, N_("Privileged register"))
P (ILL_COPROC, N_("Coprocessor error"))
P (ILL_BADSTK, N_("Internal stack error"))
#endif
#if NOW == SIGFPE
P (FPE_INTDIV, N_("Integer divide by zero"))
P (FPE_INTOVF, N_("Integer overflow"))
P (FPE_FLTDIV, N_("Floating-point divide by zero"))
P (FPE_FLTOVF, N_("Floating-point overflow"))
P (FPE_FLTUND, N_("Floating-point underflow"))
P (FPE_FLTRES, N_("Floating-poing inexact result"))
P (FPE_FLTINV, N_("Invalid floating-point operation"))
P (FPE_FLTSUB, N_("Subscript out of range"))
#endif
#if NOW == SIGSEGV
P (SEGV_MAPERR, N_("Address not mapped to object"))
P (SEGV_ACCERR, N_("Invalid permissions for mapped object"))
#endif
#if NOW == SIGBUS
P (BUS_ADRALN, N_("Invalid address alignment"))
P (BUS_ADRERR, N_("Nonexisting physical address"))
P (BUS_OBJERR, N_("Object-specific hardware error"))
#endif
#if NOW == SIGTRAP
P (TRAP_BRKPT, N_("Process breakpoint"))
P (TRAP_TRACE, N_("Process trace trap"))
#endif
#if NOW == SIGCLD
P (CLD_EXITED, N_("Child has exited"))
P (CLD_KILLED, N_("Child has terminated abnormally and did not create a core file"))
P (CLD_DUMPED, N_("Child has terminated abnormally and created a core file"))
P (CLD_TRAPPED, N_("Traced child has trapped"))
P (CLD_STOPPED, N_("Child has stopped"))
P (CLD_CONTINUED, N_("Stopped child has continued"))
#endif
#if NOW == SIGPOLL
P (POLL_IN, N_("Data input available"))
P (POLL_OUT, N_("Output buffers available"))
P (POLL_MSG, N_("Input message available"))
P (POLL_ERR, N_("I/O error"))
P (POLL_PRI, N_("High priority input available"))
P (POLL_HUP, N_("Device disconnected"))
#endif
#undef P
