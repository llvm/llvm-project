//===-- Definition of macros from sys/ioctl.h -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_LINUX_SYS_IOCTL_MACROS_H
#define LLVM_LIBC_MACROS_LINUX_SYS_IOCTL_MACROS_H

// Macros that determine command construction

#define _IOC_NRBITS 8
#define _IOC_TYPEBITS 8

#ifndef _IOC_SIZEBITS
#define _IOC_SIZEBITS 14
#endif

#ifndef _IOC_DIRBITS
#define _IOC_DIRBITS 2
#endif

#define _IOC_NRMASK ((1 << _IOC_NRBITS) - 1)
#define _IOC_TYPEMASK ((1 << _IOC_TYPEBITS) - 1)
#define _IOC_SIZEMASK ((1 << _IOC_SIZEBITS) - 1)
#define _IOC_DIRMASK ((1 << _IOC_DIRBITS) - 1)

#define _IOC_NRSHIFT 0
#define _IOC_TYPESHIFT (_IOC_NRSHIFT + _IOC_NRBITS)
#define _IOC_SIZESHIFT (_IOC_TYPESHIFT + _IOC_TYPEBITS)
#define _IOC_DIRSHIFT (_IOC_SIZESHIFT + _IOC_SIZEBITS)

#ifndef _IOC_NONE
#define _IOC_NONE 0U
#endif

#ifndef _IOC_WRITE
#define _IOC_WRITE 1U
#endif

#ifndef _IOC_READ
#define _IOC_READ 2U
#endif

// Macros for constructing commands

#define _IOC(dir, type, nr, size)                                              \
  (((dir) << _IOC_DIRSHIFT) | ((type) << _IOC_TYPESHIFT) |                     \
   ((nr) << _IOC_NRSHIFT) | ((size) << _IOC_SIZESHIFT))

#define _IO(type, nr) _IOC(_IOC_NONE, (type), (nr), 0)
#define _IOR(type, nr, argtype) _IOC(_IOC_READ, (type), (nr), sizeof(argtype))
#define _IOW(type, nr, argtype) _IOC(_IOC_WRITE, (type), (nr), sizeof(argtype))
#define _IOWR(type, nr, argtype)                                               \
  _IOC(_IOC_READ | _IOC_WRITE, (type), (nr), sizeof(argtype))

// Macros for deconstructing commands

#define _IOC_DIR(nr) (((nr) >> _IOC_DIRSHIFT) & _IOC_DIRMASK)
#define _IOC_TYPE(nr) (((nr) >> _IOC_TYPESHIFT) & _IOC_TYPEMASK)
#define _IOC_NR(nr) (((nr) >> _IOC_NRSHIFT) & _IOC_NRMASK)
#define _IOC_SIZE(nr) (((nr) >> _IOC_SIZESHIFT) & _IOC_SIZEMASK)

#define IOC_IN (_IOC_WRITE << _IOC_DIRSHIFT)
#define IOC_OUT (_IOC_READ << _IOC_DIRSHIFT)
#define IOC_INOUT ((_IOC_WRITE | _IOC_READ) << _IOC_DIRSHIFT)
#define IOCSIZE_MASK (_IOC_SIZEMASK << _IOC_SIZESHIFT)
#define IOCSIZE_SHIFT (_IOC_SIZESHIFT)

// Macros that define commands

#define TIOCPKT_DATA _IO('\0', 0)
#define TIOCSER_TEMT _IO('\0', 1)
#define TIOCPKT_FLUSHREAD TIOCSER_TEMT
#define TIOCPKT_FLUSHWRITE _IO('\0', 2)
#define TIOCPKT_STOP _IO('\0', 4)
#define TIOCPKT_START _IO('\0', 8)
#define TIOCPKT_NOSTOP _IO('\0', 16)
#define TIOCPKT_DOSTOP _IO('\0', 32)
#define TIOCPKT_IOCTL _IO('\0', 64)

#define TCGETS _IO('T', 1)
#define TCSETS _IO('T', 2)
#define TCSETSW _IO('T', 3)
#define TCSETSF _IO('T', 4)
#define TCGETA _IO('T', 5)
#define TCSETA _IO('T', 6)
#define TCSETAW _IO('T', 7)
#define TCSETAF _IO('T', 8)
#define TCSBRK _IO('T', 9)
#define TCXONC _IO('T', 10)
#define TCFLSH _IO('T', 11)
#define TIOCEXCL _IO('T', 12)
#define TIOCNXCL _IO('T', 13)
#define TIOCSCTTY _IO('T', 14)
#define TIOCGPGRP _IO('T', 15)
#define TIOCSPGRP _IO('T', 16)
#define TIOCOUTQ _IO('T', 17)
#define TIOCSTI _IO('T', 18)
#define TIOCGWINSZ _IO('T', 19)
#define TIOCSWINSZ _IO('T', 20)
#define TIOCMGET _IO('T', 21)
#define TIOCMBIS _IO('T', 22)
#define TIOCMBIC _IO('T', 23)
#define TIOCMSET _IO('T', 24)
#define TIOCGSOFTCAR _IO('T', 25)
#define TIOCSSOFTCAR _IO('T', 26)
#define FIONREAD _IO('T', 27)
#define TIOCINQ FIONREAD
#define TIOCLINUX _IO('T', 28)
#define TIOCCONS _IO('T', 29)
#define TIOCGSERIAL _IO('T', 30)
#define TIOCSSERIAL _IO('T', 31)
#define TIOCPKT _IO('T', 32)
#define FIONBIO _IO('T', 33)
#define TIOCNOTTY _IO('T', 34)
#define TIOCSETD _IO('T', 35)
#define TIOCGETD _IO('T', 36)
#define TCSBRKP _IO('T', 37)

#define TIOCSBRK _IO('T', 39)
#define TIOCCBRK _IO('T', 40)
#define TIOCGSID _IO('T', 41)

#define TIOCGRS485 _IO('T', 46)
#define TIOCSRS485 _IO('T', 47)

#define TIOCGPTN _IOR('T', 48, unsigned int)
#define TIOCSPTLCK _IOW('T', 49, int)
#define TIOCGDEV _IOR('T', 50, unsigned int)
#define TCGETX TIOCGDEV

#define TCSETX _IO('T', 51)
#define TCSETXF _IO('T', 52)
#define TCSETXW _IO('T', 53)
#define TIOCSIG _IOW('T', 54, int)
#define TIOCVHANGUP _IO('T', 55)
#define TIOCGPKT _IOR('T', 56, int)
#define TIOCGPTLCK _IOR('T', 57, int)

#define TIOCGEXCL _IOR('T', 64, int)
#define TIOCGPTPEER _IO('T', 65)

#define FIONCLEX _IO('T', 80)
#define FIOCLEX _IO('T', 81)
#define FIOASYNC _IO('T', 82)
#define TIOCSERCONFIG _IO('T', 83)
#define TIOCSERGWILD _IO('T', 84)
#define TIOCSERSWILD _IO('T', 85)
#define TIOCGLCKTRMIOS _IO('T', 86)
#define TIOCSLCKTRMIOS _IO('T', 87)
#define TIOCSERGSTRUCT _IO('T', 88)
#define TIOCSERGETLSR _IO('T', 89)
#define TIOCSERGETMULTI _IO('T', 90)
#define TIOCSERSETMULTI _IO('T', 91)
#define TIOCMIWAIT IO('T', 92)
#define TIOCGICOUNT _IO('T', 93)

#define FIOQSIZE _IO('T', 96)

#endif // LLVM_LIBC_MACROS_LINUX_SYS_IOCTL_MACROS_H
