/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 * \file
 * \brief Utility routines used by nroff-to-C utility programs.
 */

// This file was originally written in C and migrated to C++.  The first part
// of this file has remnants of old C interfaces that are still in use.
// The second part has implementations of more modern C++ interfaces.

#define USE_OLD_C_UTILS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "utils.h"

static int lineno;
static FILE *currfile = 0; /* current file being read */
static char lbuff[MAXLINE + 1];
static int currpos; /* current position within lbuff */

FILE *infile[10], *outfile[10], *savefile[10];
static FILE *null_file;
#ifdef HOST_WIN
#define NULL_FNAME "nul"
#else
#define NULL_FNAME "/dev/null"
#endif

std::list<std::string> includePaths;

inline bool
isPathSeparator(char c)
{
#ifdef _WIN64
  return  c=='\\' || c =='/'? true: false;
#else
  return c =='/'? true: false;
#endif
}

/**
 * Process command line and open input and output files. By default stdin and
 * stdout are used. Files up to "-o" are considered input files and files
 * following "-o" are output files (max of 10 each).  "-" used for a file
 * name indicates the default (stdin or stdout).  -n preceeding a file name
 * says that file is nroff output rather than C output.
 */
void
open_files(int argc, char *argv[])
{
  int i;
  const char *mode; /* file read or write mode */
  char *progname;
  FILE **fptr;
  int nsw;

  for (i = 0; i < 10; i++) {
    infile[i] = stdin;    /* default input files */
    outfile[i] = stdout;  /* default output files */
    savefile[i] = stdout; /* default output files */
  }

  if (argc > 11)
    put_err(4, "too many command line tokens");

  fptr = &infile[0];
  progname = *argv;
  {
    int ll = strlen(progname);
    char *p = progname + ll;
    while (ll--) {
      --p;
      if (*p == '.') {
        *p = '\0';                          /* skip off .exe suffix */
      } else if (isPathSeparator(*p)) { /* strip pat'h */
        progname = p + 1;
        break;
      }
    }
  }

  mode = "wb";
  null_file = fopen(NULL_FNAME, mode);
  mode = "r";

  nsw = 0;
  while (--argc > 0) {
    argv++;
    if ((*argv)[0] == '-') {
      if ((*argv)[1] == 'o') {
        /* switch from input files to output files: */
        mode = "wb";
        fptr = &outfile[0];
      } else if ((*argv)[1] == 'n')
        /* nroff output */
        nsw = 1;
      else if ((*argv)[1] == '\0')
        fptr++; /* let default file apply */
      else
        put_err(4, "illegal command line flag");
    } else {
      *fptr = fopen(*argv, mode);
      if (*fptr == NULL)
        put_err(4, "error opening file");
      if (*mode == 'w') {
        if (nsw)
          fputs(".\\\" ", *fptr);
        else
          fputs("/* ", *fptr);
        fprintf(*fptr,
                "%s - This file written by utility program %s.  Do not modify.",
                *argv, progname);
        if (nsw)
          fputc('\n', *fptr);
        else
          fputs(" */\n", *fptr);
      }
      fptr++;
      nsw = 0;
    }
  }
}

/** \brief Supress output going to output files. */
void
output_off()
{
  for (int i = 0; i < 10; i++) {
    savefile[i] = outfile[i];
    outfile[i] = null_file;
  }
}

/** \brief Enable output going to output files. */
void
output_on()
{
  for (int i = 0; i < 10; i++)
    outfile[i] = savefile[i];
}

/** \brief Same as get_line except copies to outf. */
int
get_line1(FILE *funit, LT ltypes[], FILE *outf)
{
  int i;
  static int saveline = 0;

  if (funit != currfile) { /* reading from different file than last time*/
    /*  try to maintain correct line numbers if reading from two
        input files at same time:  */
    i = lineno;
    lineno = saveline;
    saveline = i;
    currfile = funit;
  }
  currpos = 3;
  LOOP
  {
    if (fgets(lbuff, MAXLINE, funit) == NULL) {
      lineno = -1;
      return (LT_EOF);
    }
    lineno++;
    for (i = 0; ltypes[i].ltnum != 0; i++)
      if (strncmp(lbuff, ltypes[i].str, 3) == 0)
        return (ltypes[i].ltnum);
    fputs(lbuff, outf);
  }
}

void
flush_line(FILE *outf)
{
  fputs(lbuff, outf);
}

/**
   read input lines until one matching one of the line types
   in the 'ltypes' array is found.  Return the corresponding
   integer line type id, or 0 for end of file.
   Set currpos to point immediately after first 3 chars of line.
 */
int
get_line(FILE *funit, LT ltypes[])
{

  /* ltypes array of structures defining line types */

  int i;
  static int saveline = 0;

  if (funit != currfile) {
    /*  try to maintain correct line numbers if reading from two
        input files at same time:  */
    i = lineno;
    lineno = saveline;
    saveline = i;
    currfile = funit;
  }
  currpos = 3;
  LOOP
  {
    if (fgets(lbuff, MAXLINE, funit) == NULL) {
      lineno = -1;
      return (LT_EOF);
    }
    lineno++;
    for (i = 0; ltypes[i].ltnum != 0; i++) {
      if (strncmp(lbuff, ltypes[i].str, 3) == 0)
        return (ltypes[i].ltnum);
    }
  }
}

/**
   write error message to stderr with line number if non-zero
   (using global variable lineno) and severity level given by sev.
   Abort if sev == 4 (fatal error).
   \param sev error severity
   \param txt error text to be written
 */
#ifdef __cplusplus
extern "C"
#endif
void
put_err(int sev, const char *txt)
{

  static const char *sevtxt[5] = {" ", "INFORMATIONAL ", "WARNING ",
                                  "SEVERE ERROR ", "FATAL ERROR "};
  char line[200]; /* temp buffer used to construct error message */
  char tmp[10];

  if (sev < 1 || sev > 4)
    sev = 4;
  strcpy(line, sevtxt[sev]);
  if (lineno > 0) {
    strcat(line, " at line ");
    sprintf(tmp, "%d   ", lineno);
    strcat(line, tmp);
  }
  strcat(line, txt);
  strcat(line, "\n");
  fputs(line, stderr);
  if (sev == 4)
    exit(1);
}

/**
   \brief Skip past blanks in lbuff beginning at currpos, and move
   into 'out' the token up to the next blank character.

   Currpos is left pointing to the first character (blank, NULL,
   or endline) following the token.
   One of the following values is returned in len:
   - len > 0  :  number of characters in token.
   - len == 0 :  no token remaining in lbuff.
   - len == -1:  error occurred.
 */
void
get_token(char out[], int *len)
{
  int i;
  char c;
  char endtok;

  while (lbuff[currpos] == ' ')
    currpos++;
  if (lbuff[currpos] == '\0' || lbuff[currpos] == '\n') {
    *len = 0;
    return;
  }

  endtok = (lbuff[currpos] == '\"' ? '\"' : ' ');
  i = 0;
  if (endtok == '\"')
    currpos++;

  LOOP
  {
    c = lbuff[currpos];
    if (c == endtok || c == '\0' || c == '\n')
      break;
    out[i++] = c;
    currpos++;
  }

  if (endtok == '\"') {
    if (c == endtok) {
      if (i == 0)
        out[i++] = ' ';
      currpos++;
    } else
      put_err(3, "double quote missing from end of string");
  }
  out[i] = '\0';
  *len = i; /* length of token */
}

/**
   \brief dummy version of interr to satisfy external refs.
 */
#ifdef __cplusplus
extern "C"
#endif
void
interr(const char *txt, int val, int sev)
{
  char t[80];
  sprintf(t, "internal error. %s . %d", txt, val);
  put_err(sev, t);
}

//-----------------------------------------------------------------
// NroffInStream
//-----------------------------------------------------------------

void
NroffInStream::pop_file()
{
  tos().file->close();
  delete tos().file;
  stack.pop_back();
}

void
NroffInStream::close()
{
  while (!stack.empty())
    pop_file();
}

void
NroffInStream::printError(ErrorSeverity sev, const char *txt1, const char *txt2)
{
  static const char *sevtxt[] = {" ", "INFORMATIVE", "WARNING", "SEVERE ERROR",
                                 "FATAL ERROR"};
  if (sev < INFO || sev > FATAL) {
    sev = FATAL;
  }
  std::cerr << sevtxt[sev];

  // Issue file context if file is open
  if (!stack.empty()) {
    int lineno = tos().lineno;
    if (lineno > 0)
      std::cerr << " line " << lineno;
    std::cerr << " file " << tos().filename;
  }

  // Issue error message
  std::cerr << ": " << txt1 << txt2 << "\n";

  // Issue file-inclusion context
  if (stack.size() >= 2) {
    auto c = stack.end() - 1;
    do {
      --c;
      std::cerr << "included from line " << c->lineno << " file "
                << c->filename.c_str() << "\n";
    } while (c != stack.begin());
  }

  if (sev == FATAL) {
    std::exit(1);
  }
}

bool
getline(NroffInStream &f, std::string &s)
{
  while (!f.stack.empty()) {
    NroffInStream::context &c = f.stack.back();
    if (!getline(*c.file, s)) {
      f.pop_file();
    } else {
      ++c.lineno;
      if (strncmp(s.c_str(), ".so ", 4) != 0) {
        return true;
      } else {
        /* Found include directive. */
        char filename[FILENAME_MAX + 1];
        int count = sscanf(s.c_str() + 4, "%s", filename);
        if (count != 1)
          f.printError(FATAL,
                       ".so directive should be followed by a file name");
        if (strcmp(filename, "iliindex.n") == 0 ||
            strcmp(filename, "ilmindex.n") == 0) {
          /* Ignore directive - file is generated later */
          continue;
        }
        const char *outer = c.filename.c_str();
        char relFilename[FILENAME_MAX + 1];
        const char *t1 = strrchr(outer, '/');
        const char *t2 = strrchr(outer, '\\');
        const char *t = t1 > t2? t1: t2;
        if (t && isPathSeparator(*t)) {
          /* Treat include-file name as relative to directory that original
             file is in. */
          size_t k = t - outer + 1;
          memmove(relFilename + k, filename, strlen(filename) + 1);
          memcpy(relFilename, outer, k);
        }
        f.push_file(relFilename);
        /* if not found, look in the include dirs */
        for (std::list<std::string>::iterator incIter = includePaths.begin(); 
             !f && incIter != includePaths.end(); 
             incIter++) {
          std::string incFileName = *incIter + filename;
          f.push_file(incFileName.c_str());
        }
        if (!f) {
          f.printError(FATAL, "cannot open nroff include file ", filename);
        }
      }
    }
  }
  /* Reached end of outermost file. */
  return false;
}

void
collectIncludePaths(int argc, char *argv[])
{
  for (int i = 1; i < argc; i++ ) {
    if( strncmp(argv[i], "-I", 2) == 0 ) {
      char *cp = argv[i]+2;
      while (*cp && isspace(*cp)) {
        cp++;
      }
      /* On Windows we have a drive letter before the first path separator. */
#ifdef _WIN64
      if (strlen(cp) < 3 || !(isalpha(*cp) && *(cp+1) == ':' && isPathSeparator(*(cp+2)))) {
#else
      if (!isPathSeparator(*cp)) {
#endif
          put_err(3, "Include file path must be a full path name\n");
      }
      std::string inclStr(cp);
      if( !isPathSeparator(inclStr[inclStr.size()-1]) ) {
        inclStr += '/';
      }
      includePaths.push_back(inclStr);
    }
  }
#ifdef DEBUG
    std::cout << "INCLUDE FILE PATHS\n";
    for (std::list<std::string>::iterator incIter = includePaths.begin();
             incIter != includePaths.end();
             incIter++) {
      std::cout << incIter->c_ptr() <<'\n';
    }
#endif
}

//-----------------------------------------------------------------
// NroffMap
//-----------------------------------------------------------------

void
NroffMap::proxy::operator=(int code)
{
  assert(code != 0);
  assert(strlen(string) == 3);
  map->myMap[string] = code;
}

int
NroffMap::match(const std::string &line) const
{
  auto item = myMap.find(line.substr(0, 3));
  if (item != myMap.end()) {
    return item->second;
  }
  return 0;
}

const char *
NroffMap::stringOf(int code) {
  assert(code != 0);
  for (auto item = myMap.begin(); item != myMap.end(); ++item)
    if (item->second==code) 
      return item->first.c_str();
  return nullptr;
}

//-----------------------------------------------------------------
// NroffTokenStream
//-----------------------------------------------------------------

int
NroffTokenStream::get_line(const NroffMap &map)
{
  pos = nullptr;
  while (getline(ifs, buf)) {
    int result = map.match(buf);
    if (result) {
      pos = buf.c_str() + 3;
      return result;
    }
  }
  return 0;
}

bool
NroffTokenStream::get_token(std::string &tok)
{
  while (isspace(*pos))
    ++pos;

  if (!*pos)
    return false;

  // Find where token begins and ends.
  const char *begin, *end;
  if (*pos == '\"') {
    begin = pos + 1;
    end = strchr(begin, '\"');
    if (end) {
      pos = end + 1;
      if (end == begin + 1)
        printError(WARN, "double quoted string is empty");
    } else {
      printError(SEVERE, "double quote missing from end of string");
      // Assume missing " should be at end of line.
      end = strchr(begin, 0);
      pos = end;
    }
  } else {
    begin = pos;
    while (*pos && !isspace(*pos))
      ++pos;
    end = pos;
  }

  tok.assign(begin, end - begin);
  return true;
}
