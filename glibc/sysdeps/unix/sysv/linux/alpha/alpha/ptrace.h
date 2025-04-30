#ifndef __alpha_ptrace_h__
#define __alpha_ptrace_h__

/*
 * Mostly for OSF/1 compatibility.
 */

#define REG_BASE        0
#define NGP_REGS        32
#define NFP_REGS        32

#define GPR_BASE        REG_BASE
#define FPR_BASE        (GPR_BASE+NGP_REGS)
#define PC              (FPR_BASE+NFP_REGS)
#define SPR_PS          (PC+1)
#define NPTRC_REGS      (SPR_PS+1)

#endif /* __alpha_ptrace_h__ */
