//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of macros from paths.h.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_PATHS_MACROS_H
#define LLVM_LIBC_MACROS_PATHS_MACROS_H

#define _PATH_DEFPATH "/usr/bin:/bin"
#define _PATH_STDPATH "/usr/bin:/bin:/usr/sbin:/sbin"
#define _PATH_BSHELL "/bin/sh"
#define _PATH_CONSOLE "/dev/console"
#define _PATH_CSHELL "/bin/csh"
#define _PATH_DEV "/dev/"
#define _PATH_DEVNULL "/dev/null"
#define _PATH_KLOG "/proc/kmsg"
#define _PATH_MAILDIR "/var/mail"
#define _PATH_MAN "/usr/share/man"
#define _PATH_MNTTAB "/etc/fstab"
#define _PATH_MOUNTED "/etc/mtab"
#define _PATH_NOLOGIN "/etc/nologin"
#define _PATH_SENDMAIL "/usr/sbin/sendmail"
#define _PATH_SHELLS "/etc/shells"
#define _PATH_TTY "/dev/tty"
#define _PATH_TMP "/tmp/"
#define _PATH_VARDB "/var/db/"
#define _PATH_VARRUN "/var/run/"
#define _PATH_VARTMP "/var/tmp/"

#endif // LLVM_LIBC_MACROS_PATHS_MACROS_H
