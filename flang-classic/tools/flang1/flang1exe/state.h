/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file state.h
    \brief macros to read/write the 'state' file
*/

/*
 * to save/restore an array to the state file:
 *   RW_FD( address, datatype, number-elements )
 */
#define RW_FD(b, s, n)                           \
  {                                              \
    nw = (*p_rw)((char *)(b), sizeof(s), n, fd); \
    if (nw != (n))                               \
      error(10, 40, 0, "(state file)", CNULL);   \
  }

/*
 * to save/restore a scalar to the state file:
 *   RW_SCALAR( variable )
 */
#define RW_SCALAR(b) RW_FD(&b, b, 1)

/*
 * the rw_routine should be declared:
 *   void rw_routine( RW_ROUTINE, RW_FILE )
 * this declares the proper names for the rw_routine and file
 * used in the RW_ macros
 */
#define RW_ROUTINE int (*p_rw)(void *, size_t, size_t, FILE *)
#define RW_FILE FILE *fd
#define RW_ROUTINE_TYPE int (*)(void *, size_t, size_t, FILE *)

/*
 * sometimes special action is taken on read or write.
 * use these macros to test whether this is a write (save) or read (restore)
 */
#define ISREAD() (p_rw == (RW_ROUTINE_TYPE)fread)
#define ISWRITE() (p_rw == (RW_ROUTINE_TYPE)fwrite)

extern void rw_dpmout_state(RW_ROUTINE, RW_FILE);
extern void rw_semant_state(RW_ROUTINE, RW_FILE); /* semfin.c */
extern void rw_gnr_state(RW_ROUTINE, RW_FILE);    /* semgnr.c */
extern void rw_sym_state(RW_ROUTINE, RW_FILE);    /* symtab.c */
extern void rw_dtype_state(int (*p_rw)(void *, size_t, size_t, FILE *),
                           FILE *fd);             /* dtypeutl.c */
extern void rw_ast_state(RW_ROUTINE, RW_FILE);    /* ast.c */
extern void rw_dinit_state(RW_ROUTINE, RW_FILE);  /* dinit.c */
extern void rw_import_state(RW_ROUTINE, RW_FILE); /* interf.c */
extern void rw_mod_state(RW_ROUTINE, RW_FILE);    /* module.c */
