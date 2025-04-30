#ifndef _TERMIOS_H
#include <termios/termios.h>

#ifndef _ISOMAC
/* Now define the internal interfaces.  */
extern int __tcgetattr (int __fd, struct termios *__termios_p);
extern int __tcsetattr (int __fd, int __optional_actions,
			const struct termios *__termios_p);

extern int __libc_tcdrain (int __fd);

libc_hidden_proto (__tcgetattr)
libc_hidden_proto (tcsetattr)
libc_hidden_proto (cfsetispeed)
libc_hidden_proto (cfsetospeed)

#endif
#endif
