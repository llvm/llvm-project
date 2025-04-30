/* Common code for file-based database parsers in nss_files module.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <ctype.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <nss_files.h>

/* These symbols are defined by the including source file:

   ENTNAME -- database name of the structure and functions (hostent, pwent).
   STRUCTURE -- struct name, define only if not ENTNAME (passwd, group).
   DATABASE -- string of the database file's name ("hosts", "passwd").

   ENTDATA -- if defined, `struct ENTDATA' is used by the parser to store
	      things pointed to by the resultant `struct STRUCTURE'.

   NEED_H_ERRNO - defined iff an arg `int *herrnop' is used.

   EXTRA_ARGS -- defined iff extra parameters must be passed to the parser
   EXTRA_ARGS_DECL -- declaration for these extra parameters
   EXTRA_ARGS_VALUE -- values to be passed for these parameters

   Also see files-XXX.c.  */

#ifndef EXTRA_ARGS
# define EXTRA_ARGS
# define EXTRA_ARGS_DECL
# define EXTRA_ARGS_VALUE
#endif

#define CONCAT(a,b) CONCAT1(a,b)
#define CONCAT1(a,b) a##b

#ifndef STRUCTURE
# define STRUCTURE ENTNAME
#endif


struct parser_data
  {
#ifdef ENTDATA
    struct ENTDATA entdata;
# define ENTDATA_DECL(data) struct ENTDATA *const entdata = &data->entdata;
#else
# define ENTDATA_DECL(data)
#endif
    char linebuffer[0];
  };

#ifdef ENTDATA
/* The function can't be exported, because the entdata structure
   is defined only in files-foo.c.  */
# define parser_stclass static
# define nss_files_parse_hidden_def(name)
#else
/* Export the line parser function so it can be used in nss_db.  */
# define parser_stclass /* Global */
# define parse_line CONCAT(_nss_files_parse_,ENTNAME)
# define nss_files_parse_hidden_def(name) libc_hidden_def (name)
#endif


#ifdef EXTERN_PARSER

/* The parser is defined in a different module.  */
extern int parse_line (char *line, void *result,
		       struct parser_data *data, size_t datalen, int *errnop
		       EXTRA_ARGS_DECL);

# define LINE_PARSER(EOLSET, BODY) /* Do nothing */

#else

/* Define a line parsing function.  */

# define LINE_PARSER(EOLSET, BODY)					      \
parser_stclass int							      \
parse_line (char *line, void *generic_result,				      \
	    struct parser_data *data, size_t datalen, int *errnop	      \
	    EXTRA_ARGS_DECL)						      \
{									      \
  struct STRUCTURE *result = generic_result;				      \
  ENTDATA_DECL (data)							      \
  BUFFER_PREPARE							      \
  char *p = strpbrk (line, EOLSET "\n");				      \
  if (p != NULL)							      \
    *p = '\0';								      \
  BODY;									      \
  TRAILING_LIST_PARSER;							      \
  return 1;								      \
}									      \
nss_files_parse_hidden_def (parse_line)


# define STRING_FIELD(variable, terminator_p, swallow)			      \
  {									      \
    variable = line;							      \
    while (*line != '\0' && !terminator_p (*line))			      \
      ++line;								      \
    if (*line != '\0')							      \
      {									      \
	*line = '\0';							      \
	do								      \
	  ++line;							      \
	while (swallow && terminator_p (*line));			      \
      }									      \
  }

# define STRING_LIST(variable, terminator_c) \
  {									      \
    char **list = parse_list (&line, buf_start, buf_end, terminator_c,	      \
			      errnop);					      \
    if (list)								      \
      variable = list;							      \
    else								      \
      return -1;		/* -1 indicates we ran out of space.  */      \
									      \
    /* Determine the new end of the buffer.  */				      \
    while (*list != NULL)						      \
      ++list;								      \
    buf_start = (char *) (list + 1);					      \
  }

/* Helper function.  */
static inline uint32_t
__attribute__ ((always_inline))
strtou32 (const char *nptr, char **endptr, int base)
{
  unsigned long int val = strtoul (nptr, endptr, base);

  /* Match the 32-bit behavior on 64-bit platforms.  */
  if (sizeof (long int) > 4 && val > 0xffffffff)
    val = 0xffffffff;

  return val;
}

# define INT_FIELD(variable, terminator_p, swallow, base, convert)	      \
  {									      \
    char *endp;								      \
    variable = convert (strtou32 (line, &endp, base));			      \
    if (endp == line)							      \
      return 0;								      \
    else if (terminator_p (*endp))					      \
      do								      \
	++endp;								      \
      while (swallow && terminator_p (*endp));				      \
    else if (*endp != '\0')						      \
      return 0;								      \
    line = endp;							      \
  }

# define INT_FIELD_MAYBE_NULL(variable, terminator_p, swallow, base, convert, default)	      \
  {									      \
    char *endp;								      \
    if (*line == '\0')							      \
      /* We expect some more input, so don't allow the string to end here. */ \
      return 0;								      \
    variable = convert (strtou32 (line, &endp, base));			      \
    if (endp == line)							      \
      variable = default;						      \
    if (terminator_p (*endp))						      \
      do								      \
	++endp;								      \
      while (swallow && terminator_p (*endp));				      \
    else if (*endp != '\0')						      \
      return 0;								      \
    line = endp;							      \
  }

# define ISCOLON(c) ((c) == ':')


# ifndef TRAILING_LIST_MEMBER
#  define BUFFER_PREPARE /* Nothing to do.  */
#  define TRAILING_LIST_PARSER /* Nothing to do.  */
# else

# define BUFFER_PREPARE \
  char *buf_start = NULL;						      \
  char *buf_end = (char *) data + datalen;				      \
  if (line >= data->linebuffer && line < buf_end)			      \
    /* Find the end of the line buffer, we will use the space in	      \
       DATA after it for storing the vector of pointers.  */		      \
    buf_start = strchr (line, '\0') + 1;				      \
  else									      \
    /* LINE does not point within DATA->linebuffer, so that space is	      \
       not being used for scratch space right now.  We can use all of	      \
       it for the pointer vector storage.  */				      \
    buf_start = data->linebuffer;					      \

#  define TRAILING_LIST_PARSER \
{									      \
  if (buf_start == NULL)						      \
    {									      \
      if (line >= data->linebuffer && line < buf_end)			      \
	/* Find the end of the line buffer, we will use the space in	      \
	   DATA after it for storing the vector of pointers.  */	      \
	buf_start = strchr (line, '\0') + 1;				      \
      else								      \
	/* LINE does not point within DATA->linebuffer, so that space is      \
	   not being used for scratch space right now.  We can use all of     \
	   it for the pointer vector storage.  */			      \
	buf_start = data->linebuffer;					      \
    }									      \
									      \
  char **list = parse_list (&line, buf_start, buf_end, '\0', errnop);	      \
  if (list)								      \
    result->TRAILING_LIST_MEMBER = list;				      \
  else									      \
    return -1;		/* -1 indicates we ran out of space.  */	      \
}

static inline char **
__attribute ((always_inline))
parse_list (char **linep, char *eol, char *buf_end, int terminator_c,
	    int *errnop)
{
  char *line = *linep;
  char **list, **p;

  /* Adjust the pointer so it is aligned for storing pointers.  */
  eol += __alignof__ (char *) - 1;
  eol -= (eol - (char *) 0) % __alignof__ (char *);
  /* We will start the storage here for the vector of pointers.  */
  list = (char **) eol;

  p = list;
  while (1)
    {
      if ((char *) (p + 2) > buf_end)
	{
	  /* We cannot fit another pointer in the buffer.  */
	  *errnop = ERANGE;
	  return NULL;
	}

      if (*line == '\0')
	break;
      if (*line == terminator_c)
	{
	  ++line;
	  break;
	}

      /* Skip leading white space.  This might not be portable but useful.  */
      while (isspace (*line))
	++line;

      char *elt = line;
      while (1)
	{
	  if (*line == '\0' || *line == terminator_c
	      || TRAILING_LIST_SEPARATOR_P (*line))
	    {
	      /* End of the next entry.  */
	      if (line > elt)
		/* We really found some data.  */
		*p++ = elt;

	      /* Terminate string if necessary.  */
	      if (*line != '\0')
		{
		  char endc = *line;
		  *line++ = '\0';
		  if (endc == terminator_c)
		    goto out;
		}
	      break;
	    }
	  ++line;
	}
    }
 out:
  *p = NULL;
  *linep = line;

  return list;
}

# endif	/* TRAILING_LIST_MEMBER */
#endif	/* EXTERN_PARSER */


#define LOOKUP_NAME(nameelt, aliaselt)					      \
{									      \
  char **ap;								      \
  if (! strcmp (name, result->nameelt))					      \
    break;								      \
  for (ap = result->aliaselt; *ap; ++ap)				      \
    if (! strcmp (name, *ap))						      \
      break;								      \
  if (*ap)								      \
    break;								      \
}

#define LOOKUP_NAME_CASE(nameelt, aliaselt)				      \
{									      \
  char **ap;								      \
  if (! __strcasecmp (name, result->nameelt))				      \
    break;								      \
  for (ap = result->aliaselt; *ap; ++ap)				      \
    if (! __strcasecmp (name, *ap))					      \
      break;								      \
  if (*ap)								      \
    break;								      \
}


/* This is defined by db-*.c to include "../nss_db/db-XXX.c" instead.  */
#ifndef GENERIC
# define GENERIC "files-XXX.c"
#endif
