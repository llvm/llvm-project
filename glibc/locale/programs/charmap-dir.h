/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published
   by the Free Software Foundation; version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#ifndef _CHARMAP_DIR_H
#define _CHARMAP_DIR_H 1

/* The data type of a charmap directory being traversed.  */
typedef struct charmap_dir CHARMAP_DIR;

/* Starts a charmap directory traversal.
   Returns a CHARMAP_DIR, or NULL if the directory doesn't exist.  */
extern CHARMAP_DIR *charmap_opendir (const char *directory);

/* Reads the next directory entry.
   Returns its charmap name, or NULL if past the last entry or upon error.
   The storage returned may be overwritten by a later charmap_readdir
   call on the same CHARMAP_DIR.  */
extern const char *charmap_readdir (CHARMAP_DIR *dir);

/* Finishes a charmap directory traversal, and frees the resources
   attached to the CHARMAP_DIR.  */
extern int charmap_closedir (CHARMAP_DIR *dir);

/* Returns a NULL terminated list of alias names of a charmap.  */
extern char **charmap_aliases (const char *directory, const char *name);

/* Frees an alias list returned by charmap_aliases.  */
extern void charmap_free_aliases (char **aliases);

/* Opens a charmap for reading, given its name (not an alias name).  */
extern FILE *charmap_open (const char *directory, const char *name);

#endif /* _CHARMAP_DIR_H */
