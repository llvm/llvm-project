//===-- Definition of macros from ftw.h -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_FTW_MACROS_H
#define LLVM_LIBC_MACROS_FTW_MACROS_H

// Values for the typeflag argument to the callback function.
#define FTW_F 0   // Non-directory file.
#define FTW_D 1   // Directory.
#define FTW_DNR 2 // Directory without read permission.
#define FTW_NS 3  // Unknown type; stat() failed.
#define FTW_SL 4  // Symbolic link.
#define FTW_DP 5  // Directory with subdirectories visited.
#define FTW_SLN 6 // Symbolic link naming non-existing file.

// Flags for the flags argument to nftw().
#define FTW_PHYS 1  // Physical walk, does not follow symbolic links.
#define FTW_MOUNT 2 // The walk does not cross a mount point.
#define FTW_CHDIR 4 // Change to each directory before processing.
#define FTW_DEPTH 8 // All subdirectories visited before the directory.

#ifdef _GNU_SOURCE
#define FTW_ACTIONRETVAL 16 // Use FTW_* action return values (GNU extension).

// Return values from callback functions (when FTW_ACTIONRETVAL is set).
#define FTW_CONTINUE 0      // Continue with next sibling.
#define FTW_STOP 1          // Return from ftw/nftw with FTW_STOP.
#define FTW_SKIP_SUBTREE 2  // Don't walk through subtree (FTW_D only).
#define FTW_SKIP_SIBLINGS 3 // Skip siblings, continue with parent.
#endif                      // _GNU_SOURCE

#endif // LLVM_LIBC_MACROS_FTW_MACROS_H
