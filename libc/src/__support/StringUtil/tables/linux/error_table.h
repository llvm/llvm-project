//===-- Map from error numbers to strings on linux --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_LINUX_ERROR_TABLE_H
#define LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_LINUX_ERROR_TABLE_H

#include "src/errno/libc_errno.h" // For error macros

#include "src/__support/CPP/array.h"
#include "src/__support/StringUtil/message_mapper.h"

namespace __llvm_libc::internal {

constexpr MsgTable<52> LINUX_ERRORS = {
    MsgMapping(ENOTBLK, "Block device required"),
    MsgMapping(ECHRNG, "Channel number out of range"),
    MsgMapping(EL2NSYNC, "Level 2 not synchronized"),
    MsgMapping(EL3HLT, "Level 3 halted"),
    MsgMapping(EL3RST, "Level 3 reset"),
    MsgMapping(ELNRNG, "Link number out of range"),
    MsgMapping(EUNATCH, "Protocol driver not attached"),
    MsgMapping(ENOCSI, "No CSI structure available"),
    MsgMapping(EL2HLT, "Level 2 halted"),
    MsgMapping(EBADE, "Invalid exchange"),
    MsgMapping(EBADR, "Invalid request descriptor"),
    MsgMapping(EXFULL, "Exchange full"),
    MsgMapping(ENOANO, "No anode"),
    MsgMapping(EBADRQC, "Invalid request code"),
    MsgMapping(EBADSLT, "Invalid slot"),
    MsgMapping(EBFONT, "Bad font file format"),
    MsgMapping(ENONET, "Machine is not on the network"),
    MsgMapping(ENOPKG, "Package not installed"),
    MsgMapping(EREMOTE, "Object is remote"),
    MsgMapping(EADV, "Advertise error"),
    MsgMapping(ESRMNT, "Srmount error"),
    MsgMapping(ECOMM, "Communication error on send"),
    MsgMapping(EDOTDOT, "RFS specific error"),
    MsgMapping(ENOTUNIQ, "Name not unique on network"),
    MsgMapping(EBADFD, "File descriptor in bad state"),
    MsgMapping(EREMCHG, "Remote address changed"),
    MsgMapping(ELIBACC, "Can not access a needed shared library"),
    MsgMapping(ELIBBAD, "Accessing a corrupted shared library"),
    MsgMapping(ELIBSCN, ".lib section in a.out corrupted"),
    MsgMapping(ELIBMAX, "Attempting to link in too many shared libraries"),
    MsgMapping(ELIBEXEC, "Cannot exec a shared library directly"),
    MsgMapping(ERESTART, "Interrupted system call should be restarted"),
    MsgMapping(ESTRPIPE, "Streams pipe error"),
    MsgMapping(EUSERS, "Too many users"),
    MsgMapping(ESOCKTNOSUPPORT, "Socket type not supported"),
    MsgMapping(EPFNOSUPPORT, "Protocol family not supported"),
    MsgMapping(ESHUTDOWN, "Cannot send after transport endpoint shutdown"),
    MsgMapping(ETOOMANYREFS, "Too many references: cannot splice"),
    MsgMapping(EHOSTDOWN, "Host is down"),
    MsgMapping(EUCLEAN, "Structure needs cleaning"),
    MsgMapping(ENOTNAM, "Not a XENIX named type file"),
    MsgMapping(ENAVAIL, "No XENIX semaphores available"),
    MsgMapping(EISNAM, "Is a named type file"),
    MsgMapping(EREMOTEIO, "Remote I/O error"),
    MsgMapping(ENOMEDIUM, "No medium found"),
    MsgMapping(EMEDIUMTYPE, "Wrong medium type"),
    MsgMapping(ENOKEY, "Required key not available"),
    MsgMapping(EKEYEXPIRED, "Key has expired"),
    MsgMapping(EKEYREVOKED, "Key has been revoked"),
    MsgMapping(EKEYREJECTED, "Key was rejected by service"),
    MsgMapping(ERFKILL, "Operation not possible due to RF-kill"),
    MsgMapping(EHWPOISON, "Memory page has hardware error"),
};

} // namespace __llvm_libc::internal

#endif // LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_LINUX_ERROR_TABLE_H
