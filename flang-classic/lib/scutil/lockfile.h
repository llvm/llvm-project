/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file

   Four routines to be able to lock access to a file in the Linux
   approved manner.  This seems to be the only method that works
   over NFS.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
   \brief Creates a uniquely-named file in the current directory.

   To work, this must be created on the same filesystem as the file which we are
   attempting to lock.  If there are multiple processes running each competing
   for the same lock, each gets a unique file here.
 */
int __pg_make_lock_file(char *dir);

/**
   \brief The argument is the name of the lock.

   Each process tries to create a hard link with this name to its own
   uniquely-named file from __pg_make_lock_file().  The one that succeeds is the
   new lock owner.  The others fail and try again.  There is a fail-over to
   handle the case where the process with the lock dies, which is inherently
   unsafe, but we haven't come up with a better solution.
 */
void __pg_get_lock(char *lname);

/**
   \brief The argument is the same name for the lock.

   The lock is released by deleting (calling unlink) for the hard link we had
   just created.
 */
void __pg_release_lock(char *lname);

/**
   \brief Clean up by deleting the uniquely named file we had created earlier.

   These routines only allow one lock to be managed at a time.  They dynamically
   allocate and free memory.
 */
void __pg_delete_lock_file(void);

#ifdef __cplusplus
}
#endif
