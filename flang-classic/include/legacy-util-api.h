/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file
 * \brief Legacy utility interfaces
 *
 *  This header comprises declarations and definitions from the original
 *  scutil/hammer/linux86-64/include/scutil.h header file that don't
 *  pertain to constant representation or the compile-time evaluation
 *  of operations.
 */

#ifndef SCUTIL_UTIL_API_H_
#define SCUTIL_UTIL_API_H_
#ifdef __cplusplus
extern "C" {
#endif

/*
 *  TODO: "include what you use" these headers directly where needed
 *  instead of bundling them all here.
 */

#include <ctype.h>
#include <limits.h> /* PATH_MAX */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> /* time() */
#ifndef _WIN64
#include <unistd.h> /* getcwd() */
#else
#include <direct.h>
#endif

/* Copy to 'basename' the final path component, less any undesirable suffix. */
void basenam(const char *orig_path, const char *optional_suffix,
             char *basename);

/* Strip off anything after the last '/', but keep the '/' (unlike dirname(1)).
 * Return "./" if there's no '/'.
 */
void dirnam(const char *orig_path, char *dirname);

/* Locate 'file' in a list of directories, like the shell's $PATH.
 * Write the successful pathname to 'path'.  Returns 0 on success, -1 else.
 */
int fndpath(const char *file, char *path, size_t max_length,
            const char *dirs);

/* Directory name separator for fndpath: */
#define DIRSEP		':'
/* Directory names for fndpath:  */
#define	DIRWORK		""
#define	DIRSINCS	"/usr/include"

/* If 'pattern' ends in 'oldext', replace that suffix in place with 'newext'.
 * Always returns 'pattern'.
 */
char *mkperm(char *pattern, const char *oldext, const char *newext);

/* Predefined file name extensions for mkperm():  */
#define	UNKFILE		""		/* No extension. */
#define	LISTFILE	".lst"		/* Listing file. */
#define	OBJFILE		".o"		/* Object module file. */
#define	IMGFILE		".out"		/* Image module file. */
#define	SYMFILE		".dbg"		/* Symbol file. */
#define	OVDFILE		".ovd"		/* Overlay description file. */
#define	OBJLFILE	".a"		/* Object library file. */
#define	IMGLFILE	".ilb"		/* Image library file. */
#define	MAPFILE		".map"		/* Map file from linker. */
#define CFILE		".c"		/* C source */
#define IFILE		".i"		/* C .i file */
#define FTNFILE		".f"		/* Fortran source */
#define ASMFILE		".s"		/* asm source */

/* Measures user+system CPU milliseconds that elapse between calls. */
unsigned long get_rutime(void);

#ifdef __cplusplus
}
#endif
#endif /* SCUTIL_UTIL_API_H_ */
