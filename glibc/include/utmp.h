#ifndef _UTMP_H
#include <login/utmp.h>

# ifndef _ISOMAC

/* Now define the internal interfaces.  */
extern void __updwtmp (const char *__wtmp_file, const struct utmp *__utmp);
libc_hidden_proto (__updwtmp)
extern int __utmpname (const char *__file);
libc_hidden_proto (__utmpname)
extern struct utmp *__getutent (void);
libc_hidden_proto (__getutent)
extern void __setutent (void);
libc_hidden_proto (__setutent)
extern void __endutent (void);
libc_hidden_proto (__endutent)
extern struct utmp *__getutid (const struct utmp *__id);
libc_hidden_proto (__getutid)
extern struct utmp *__getutline (const struct utmp *__line);
libc_hidden_proto (__getutline)
extern struct utmp *__pututline (const struct utmp *__utmp_ptr);
libc_hidden_proto (__pututline)
extern int __getutent_r (struct utmp *__buffer, struct utmp **__result);
libc_hidden_proto (__getutent_r)
extern int __getutid_r (const struct utmp *__id, struct utmp *__buffer,
			struct utmp **__result);
libc_hidden_proto (__getutid_r)
extern int __getutline_r (const struct utmp *__line,
			  struct utmp *__buffer, struct utmp **__result);
libc_hidden_proto (__getutline_r)

libc_hidden_proto (login)
libc_hidden_proto (login_tty)
libc_hidden_proto (logout)
libc_hidden_proto (logwtmp)

# endif /* !_ISOMAC */
#endif
