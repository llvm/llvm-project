/* Check system header files for ISO 9899:1990 (ISO C) compliance.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jens Schweikhardt <schweikh@noc.dfn.de>, 1996.

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

/* This is a simple minded program that tries to find illegal macro
   definitions in system header files. Illegal macro definitions are
   those not from the implementation namespace (i.e. not starting with
   an underscore) or not matching any identifier mandated by The
   Standard. Some common macro names are considered okay, e.g. all those
   beginning with E (which may be defined in <errno.h>) or ending in
   _MAX. See the arrays prefix[] and suffix[] below for details.

   In a compliant implementation no other macros can be defined, because
   you could write strictly conforming programs that may fail to compile
   due to syntax errors: suppose <stdio.h> defines PIPE_BUF, then the
   conforming

   #include <assert.h>
   #include <stdio.h>      <- or where the bogus macro is defined
   #include <string.h>
   #define STR(x) #x
   #define XSTR(x) STR(x)
   int main (void)
   {
     int PIPE_BUF = 0;
     assert (strcmp ("PIPE_BUF", XSTR (PIPE_BUF)) == 0);
     return 0;
   }

   is expected to compile and meet the assertion. If it does not, your
   compiler compiles some other language than Standard C.

   REQUIREMENTS:
     This program calls gcc to get the list of defined macros. If you
     don't have gcc you're probably out of luck unless your compiler or
     preprocessor has something similar to gcc's -dM option. Tune
     PRINT_MACROS in this case. This program assumes headers are found
     under /usr/include and that there is a writable /tmp directory.
     Tune SYSTEM_INCLUDE if your system differs.
     #define BROKEN_SYSTEM if system(NULL) bombs -- one more violation
     of ISO C, by the way.

   OUTPUT:
     Each header file name is printed, followed by illegal macro names
     and their definition. For the above example, you would see
     ...
     /usr/include/stdio.h
     #define PIPE_BUF 5120
     ...
     If your implementation does not yet incorporate Amendment 1 you
     will see messages about iso646.h, wctype.h and wchar.h not being
     found.  */

#ifndef _GNU_SOURCE
# define _GNU_SOURCE 1
#endif

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define HEADER_MAX          256

static char macrofile[] = "/tmp/isomac.XXXXXX";

/* ISO C header names including Amendment 1 (without ".h" suffix).  */
static char *header[] =
{
  "assert", "ctype", "errno", "float", "iso646", "limits", "locale",
  "math", "setjmp", "signal", "stdarg", "stddef", "stdio", "stdlib",
  "string", "time", "wchar", "wctype"
};

/* Macros with these prefixes are considered okay.  */
static char *prefix[] =
{
  "_", "E", "is", "str", "mem", "SIG", "FLT_", "DBL_", "LDBL_",
  "LC_", "wmem", "wcs"
};

/* Macros with these suffixes are considered okay.  Will not work for
   parametrized macros with arguments.  */
static char *suffix[] =
{
  "_MAX", "_MIN"
};

/* These macros are considered okay. In fact, these are just more prefixes.  */
static char *macros[] =
{
  "BUFSIZ", "CHAR_BIT", "CHAR_MAX", "CHAR_MIN", "CLOCKS_PER_SEC",
  "DBL_DIG", "DBL_EPSILON", "DBL_MANT_DIG", "DBL_MAX",
  "DBL_MAX_10_EXP", "DBL_MAX_EXP", "DBL_MIN", "DBL_MIN_10_EXP",
  "DBL_MIN_EXP", "EDOM", "EILSEQ", "EOF", "ERANGE", "EXIT_FAILURE",
  "EXIT_SUCCESS", "FILENAME_MAX", "FLT_DIG", "FLT_EPSILON",
  "FLT_MANT_DIG", "FLT_MAX", "FLT_MAX_10_EXP", "FLT_MAX_EXP",
  "FLT_MIN", "FLT_MIN_10_EXP", "FLT_MIN_EXP", "FLT_RADIX",
  "FLT_ROUNDS", "FOPEN_MAX", "HUGE_VAL", "INT_MAX", "INT_MIN",
  "LC_ALL", "LC_COLLATE", "LC_CTYPE", "LC_MONETARY", "LC_NUMERIC",
  "LC_TIME", "LDBL_DIG", "LDBL_EPSILON", "LDBL_MANT_DIG", "LDBL_MAX",
  "LDBL_MAX_10_EXP", "LDBL_MAX_EXP", "LDBL_MIN", "LDBL_MIN_10_EXP",
  "LDBL_MIN_EXP", "LONG_MAX", "LONG_MIN", "L_tmpnam", "MB_CUR_MAX",
  "MB_LEN_MAX", "NDEBUG", "NULL", "RAND_MAX", "SCHAR_MAX",
  "SCHAR_MIN", "SEEK_CUR", "SEEK_END", "SEEK_SET", "SHRT_MAX",
  "SHRT_MIN", "SIGABRT", "SIGFPE", "SIGILL", "SIGINT", "SIGSEGV",
  "SIGTERM", "SIG_DFL", "SIG_ERR", "SIG_IGN", "TMP_MAX", "UCHAR_MAX",
  "UINT_MAX", "ULONG_MAX", "USHRT_MAX", "WCHAR_MAX", "WCHAR_MIN",
  "WEOF", "_IOFBF", "_IOLBF", "_IONBF", "abort", "abs", "acos",
  "acosf", "acosl", "and", "and_eq", "asctime", "asin", "asinf",
  "asinl", "assert", "atan", "atan2", "atan2f", "atan2l", "atanf",
  "atanl", "atexit", "atof", "atoi", "atol", "bitand", "bitor",
  "bsearch", "btowc", "calloc", "ceil", "ceilf", "ceill", "clearerr",
  "clock", "clock_t", "compl", "cos", "cosf", "cosh", "coshf",
  "coshl", "cosl", "ctime", "difftime", "div", "div_t", "errno",
  "exit", "exp", "expf", "expl", "fabs", "fabsf", "fabsl", "fclose",
  "feof", "ferror", "fflush", "fgetc", "fgetpos", "fgets", "fgetwc",
  "fgetws", "floor", "floorf", "floorl", "fmod", "fmodf", "fmodl",
  "fopen", "fprintf", "fputc", "fputs", "fputwc", "fputws", "fread",
  "free", "freopen", "frexp", "frexpf", "frexpl", "fscanf", "fseek",
  "fsetpos", "ftell", "fwide", "fwprintf", "fwrite", "fwscanf",
  "getc", "getchar", "getenv", "gets", "getwc", "getwchar", "gmtime",
  "isalnum", "isalpha", "iscntrl", "isdigit", "isgraph", "islower",
  "isprint", "ispunct", "isspace", "isupper", "iswalnum", "iswalpha",
  "iswcntrl", "iswctype", "iswdigit", "iswgraph", "iswlower",
  "iswprint", "iswpunct", "iswspace", "iswupper", "iswxdigit",
  "isxdigit", "labs", "ldexp", "ldexpf", "ldexpl", "ldiv", "ldiv_t",
  "localeconv", "localtime", "log", "log10", "log10f", "log10l",
  "logf", "logl", "longjmp", "malloc", "mblen", "mbrlen", "mbrtowc",
  "mbsinit", "mbsrtowcs", "mbstate_t", "mbstowcs", "mbtowc", "memchr",
  "memcmp", "memcpy", "memmove", "memset", "mktime", "modf", "modff",
  "modfl", "not", "not_eq", "offsetof", "or", "or_eq", "perror",
  "pow", "powf", "powl", "printf", "ptrdiff_t", "putc", "putchar",
  "puts", "putwc", "putwchar", "qsort", "raise", "rand", "realloc",
  "remove", "rename", "rewind", "scanf", "setbuf", "setjmp",
  "setlocale", "setvbuf", "sig_atomic_t", "signal", "sin", "sinf",
  "sinh", "sinhf", "sinhl", "sinl", "size_t", "sprintf", "sqrt",
  "sqrtf", "sqrtl", "srand", "sscanf", "stderr", "stdin", "stdout",
  "strcat", "strchr", "strcmp", "strcoll", "strcpy", "strcspn",
  "strerror", "strftime", "strlen", "strncat", "strncmp", "strncpy",
  "strpbrk", "strrchr", "strspn", "strstr", "strtod", "strtok",
  "strtol", "strtoul", "strxfrm", "swprintf", "swscanf", "system",
  "tan", "tanf", "tanh", "tanhf", "tanhl", "tanl", "time", "time_t",
  "tmpfile", "tmpnam", "tolower", "toupper", "towctrans", "towlower",
  "towupper", "ungetc", "ungetwc", "va_arg", "va_copy", "va_end", "va_start",
  "vfprintf", "vfwprintf", "vprintf", "vsprintf", "vswprintf",
  "vwprintf", "wchar_t", "wcrtomb", "wcscat", "wcschr", "wcscmp",
  "wcscoll", "wcscpy", "wcscspn", "wcsftime", "wcslen", "wcsncat",
  "wcsncmp", "wcsncpy", "wcspbrk", "wcsrchr", "wcsrtombs", "wcsspn",
  "wcsstr", "wcstod", "wcstok", "wcstol", "wcstombs", "wcstoul",
  "wcsxfrm", "wctob", "wctomb", "wctrans", "wctrans_t", "wctype",
  "wctype_t", "wint_t", "wmemchr", "wmemcmp", "wmemcpy", "wmemmove",
  "wmemset", "wprintf", "wscanf", "xor", "xor_eq"
};

#define NUMBER_OF_HEADERS              (sizeof header / sizeof *header)
#define NUMBER_OF_PREFIXES             (sizeof prefix / sizeof *prefix)
#define NUMBER_OF_SUFFIXES             (sizeof suffix / sizeof *suffix)
#define NUMBER_OF_MACROS               (sizeof macros / sizeof *macros)


/* Format string to build command to invoke compiler.  */
static const char fmt[] = "\
echo \"#include <%s>\" |\
%s -E -dM -ansi -pedantic %s -D_LIBC -D_ISOMAC \
-DIN_MODULE=MODULE_extramodules -I. \
-isystem `%s --print-prog-name=include` - 2> /dev/null > %s";


/* The compiler we use (given on the command line).  */
char *CC;
/* The -I parameters for CC to find all headers.  */
char *INC;

static char *xstrndup (const char *, size_t);
static const char **get_null_defines (void);
static int check_header (const char *, const char **);

int
main (int argc, char *argv[])
{
  int h;
  int result = 0;
  const char **ignore_list;

  CC = argc > 1 ? argv[1] : "gcc";
  INC = argc > 2 ? argv[2] : "";

  if (system (NULL) == 0)
    {
      puts ("Sorry, no command processor.");
      return EXIT_FAILURE;
    }

  /* First get list of symbols which are defined by the compiler.  */
  ignore_list = get_null_defines ();

  fputs ("Tested files:\n", stdout);

  for (h = 0; h < NUMBER_OF_HEADERS; ++h)
    {
      char file_name[HEADER_MAX];
      sprintf (file_name, "%s.h", header[h]);
      result |= check_header (file_name, ignore_list);
    }

  remove (macrofile);

  /* The test suite should return errors but for now this is not
     practical.  Give a warning and ask the user to correct the bugs.  */
  return result;
}


static char *
xstrndup (const char *s, size_t n)
{
  size_t len = n;
  char *new = malloc (len + 1);

  if (new == NULL)
    return NULL;

  new[len] = '\0';
  return memcpy (new, s, len);
}


static const char **
get_null_defines (void)
{
  char line[BUFSIZ], *command;
  char **result = NULL;
  size_t result_len = 0;
  size_t result_max = 0;
  FILE *input;
  int first = 1;

  int fd = mkstemp (macrofile);
  if (fd == -1)
    {
      printf ("mkstemp failed: %m\n");
      exit (1);
    }
  close (fd);

  command = malloc (sizeof fmt + sizeof "/dev/null" + 2 * strlen (CC)
		    + strlen (INC) + strlen (macrofile));

  if (command == NULL)
    {
      puts ("No more memory.");
      exit (1);
    }

  sprintf (command, fmt, "/dev/null", CC, INC, CC, macrofile);

  if (system (command))
    {
      puts ("system() returned nonzero");
      free (command);
      return NULL;
    }
  free (command);
  input = fopen (macrofile, "r");

  if (input == NULL)
    {
      printf ("Could not read %s: ", macrofile);
      perror (NULL);
      return NULL;
    }

  while (fgets (line, sizeof line, input) != NULL)
    {
      int i, okay = 0;
      size_t endmac;
      char *start, *end;
      if (strlen (line) < 9 || line[7] != ' ')
	{ /* "#define A" */
	  printf ("Malformed input, expected '#define MACRO'\ngot '%s'\n",
		  line);
	  continue;
	}
      if (line[8] == '_')
	/* It's a safe identifier.  */
	continue;
      if (result_len == result_max)
	{
	  result_max += 10;
	  result = realloc (result, result_max * sizeof (char **));
	  if (result == NULL)
	    {
	      puts ("No more memory.");
	      exit (1);
	    }
	}
      start = &line[8];
      for (end = start + 1; !isspace (*end) && *end != '\0'; ++end)
	;
      result[result_len] = xstrndup (start, end - start);

      if (strcmp (result[result_len], "IN_MODULE") != 0)
	{
	  if (first)
	    {
	      fputs ("The following identifiers will be ignored since the compiler defines them\nby default:\n", stdout);
	      first = 0;
	    }
	  puts (result[result_len]);
	}
      ++result_len;
    }
  if (result_len == result_max)
    {
      result_max += 1;
      result = realloc (result, result_max * sizeof (char **));
      if (result == NULL)
	{
	  puts ("No more memory.");
	  exit (1);
	}
    }
  result[result_len] = NULL;
  fclose (input);

  return (const char **) result;
}


static int
check_header (const char *file_name, const char **except)
{
  char line[BUFSIZ], *command;
  FILE *input;
  int result = 0;

  command = malloc (sizeof fmt + strlen (file_name) + 2 * strlen (CC)
		    + strlen (INC) + strlen (macrofile));

  if (command == NULL)
    {
      puts ("No more memory.");
      exit (1);
    }

  puts (file_name);
  sprintf (command, fmt, file_name, CC, INC, CC, macrofile);

  if (system (command))
    {
      puts ("system() returned nonzero");
      result = 1;
    }
  free (command);
  input = fopen (macrofile, "r");

  if (input == NULL)
    {
      printf ("Could not read %s: ", macrofile);
      perror (NULL);
      return 1;
    }

  while (fgets (line, sizeof line, input) != NULL)
    {
      int i, okay = 0;
      size_t endmac;
      const char **cpp;
      if (strlen (line) < 9 || line[7] != ' ')
	{ /* "#define A" */
	  printf ("Malformed input, expected '#define MACRO'\ngot '%s'\n",
		  line);
	  result = 1;
	  continue;
	}
      for (i = 0; i < NUMBER_OF_PREFIXES; ++i)
	{
	  if (!strncmp (line+8, prefix[i], strlen (prefix[i]))) {
	    ++okay;
	    break;
	  }
	}
      if (okay)
	continue;
      for (i = 0; i < NUMBER_OF_MACROS; ++i)
	{
	  if (!strncmp (line + 8, macros[i], strlen (macros[i])))
	    {
	      ++okay;
	      break;
	    }
	}
      if (okay)
	continue;
      /* Find next char after the macro identifier; this can be either
	 a space or an open parenthesis.  */
      endmac = strcspn (line + 8, " (");
      if (line[8+endmac] == '\0')
	{
	  printf ("malformed input, expected '#define MACRO VALUE'\n"
		  "got '%s'\n", line);
	  result = 1;
	  continue;
	}
      for (i = 0; i < NUMBER_OF_SUFFIXES; ++i)
	{
	  size_t len = strlen (suffix[i]);
	  if (!strncmp (line + 8 + endmac - len, suffix[i], len))
	    {
	      ++okay;
	      break;
	    }
	}
      if (okay)
	continue;
      if (except != NULL)
	for (cpp = except; *cpp != NULL; ++cpp)
	  {
	    size_t len = strlen (*cpp);
	    if (!strncmp (line + 8, *cpp, len) && isspace (line[8 + len]))
	      {
		++okay;
		break;
	      }
	  }
      if (!okay)
	{
	  fputs (line, stdout);
	  result = 2;
	}
    }
  fclose (input);

  return result;
}

/* EOF */
