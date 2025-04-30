/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

static char linbuf[121];
static char savelin[121];
static char name[60];
static char case_name[60];
static FILE *infile, *outfile, *prodfile;

static int curchr, fstchr, lineno;
static int current;

static int bcnt = 2;

static void readln();
static void header();
static void footer();
static int scan();
static void error(char *);
static void init();
static void output();

#define TOK_ID 0
#define TOK_EQUAL 1
#define TOK_OR 2
#define TOK_EOLN 3
#define TOK_EOF 4
#define TOK_BREAK 5
#define TOK_OTHER 6

static const char *tok_name[] = {"id",  "equal", "or",   "eoln",
                                 "eof", "break", "other"};

int
main()
{
  int i;
  int tok;

  infile = fopen("gram.txt", "r");
  if (infile == NULL) {
    printf("Can't open input file\n");
    exit(0);
  }
  outfile = fopen("actions", "w");
  if (outfile == NULL) {
    printf("Can't open output file\n");
    exit(0);
  }
  prodfile = fopen("proddf.h", "w");
  if (prodfile == NULL) {
    printf("Can't open prodstr file\n");
    exit(0);
  }

  linbuf[0] = '\0';
  header();
  readln();
  while (1) {
    tok = scan();
    if (tok == TOK_ID) {
      i = curchr - fstchr;
      strncpy(name, &linbuf[fstchr], i);
      name[i] = '\0';
      tok = scan();
      if (tok != TOK_EQUAL) {
        error("equal expected");
        goto close;
      }
      init();
      while (1) {
        tok = scan();
        if (tok == TOK_OR) {
          output();
        } else if (tok == TOK_EOLN) {
          output();
          break;
        } else if (tok == TOK_EOF) {
          error("unexpected eof");
          goto close;
        }
      }
    } else if (tok == TOK_EOF)
      break;
    else if (tok == TOK_BREAK) {
      /* output(); */
      fprintf(outfile, "    }\n\n");
      current = 0;
      fprintf(outfile, "/*****  SEMANTIC BREAK - SEM%d *****/\n\n", bcnt);
      fprintf(outfile, "    switch(rednum) {\n");
      ++bcnt;
    } else if (tok != TOK_EOLN) {
      printf("illegal token %s, line %d\n", tok_name[tok], lineno);
      break;
    }
  }
  footer();

close:
  fclose(infile);
  fclose(outfile);
  fclose(prodfile);
  return 0;
}

static void
readln()
{
  char *status;

  strncpy(savelin, linbuf, 121);
  savelin[strlen(savelin) - 1] = '\0';
  status = fgets(linbuf, 121, infile);
  lineno++;
  curchr = 0;
  if (status == NULL) {
    linbuf[0] = '\0';
  }
  return;
}

static int
scan()
{
  int token = -1;
  char c;

  while (1) {
    c = linbuf[curchr];
    fstchr = curchr;
    curchr++;
    if (c == ' ' || c == '\t')
      continue;
    if (c == '<') {
      while (linbuf[curchr] != '>' && linbuf[curchr] != '\n')
        curchr++;
      if (linbuf[curchr] == '\n')
        error("unmatched left angle bracket");
      curchr++;
      token = TOK_ID;
    } else if (c == ':' && linbuf[curchr] == ':' && linbuf[curchr + 1] == '=') {
      curchr = curchr + 2;
      token = TOK_EQUAL;
    } else if (c == '|') {
      token = TOK_OR;
      if (linbuf[curchr] != '\n') {
        error("warning: | does not end line");
      }
      readln();
    } else if (c == '\'') {
      while (linbuf[curchr] != '\'' && linbuf[curchr] != '\n')
        curchr++;
      if (linbuf[curchr] == '\n')
        error("unmatched single quote");
      curchr++;
      continue;
    } else if (c == '\n') {
      readln();
      token = TOK_EOLN;
    } else if (c == '\0') {
      token = TOK_EOF;
    } else if (c == '_') {
      readln();
      continue;
    } else if (c == '>') {
      error("isolated right angle bracket");
    } else if (fstchr == 0 && linbuf[0] == '.' && linbuf[1] == 'B') {
      readln();
      token = TOK_BREAK;
    } else if (linbuf[0] == '#') {
        readln();
        continue;
    } else {
      while (linbuf[curchr] != ' ' && linbuf[curchr] != '\t' &&
             linbuf[curchr] != '\n')
        curchr++;
      token = TOK_OTHER;
    }
    break;
  }
  return token;
}

static void error(char *p)
{
  printf("%s, line %d\n", p, lineno);
  return;
}

static void
header()
{
  fprintf(outfile, "  switch(rednum) {\n\n");
  fprintf(outfile, "  /* "
                   "-----------------------------------------------------------"
                   "------- */");
  fprintf(outfile, "\n  /*\n   *\t<SYSTEM GOAL SYMBOL> ::=\n   */\n");
  fprintf(outfile, "  case SYSTEM_GOAL_SYMBOL1:\n");
  fprintf(outfile, "    break;\n");

  fprintf(prodfile, "static const char *prodstr[] = {\"SYSTEM GOAL\",\n");

  return;
}

static void
footer()
{
  fprintf(outfile, "  }\n");

  fprintf(prodfile, "};\n");

  return;
}

static void
init()
{
  char c;
  char *p, *q;
  int i;

  current = 0;
  p = name + 1;
  q = case_name;
  i = 0;
  /*
      while(1) {
          c = *p++;
          if (c == ' ' || c == '\t') {
              while ( (c = *p++) == ' ' || c == '\t' ) ;
              break;
          }
          if (c == '>') break;
          *q++ = (isupper(c)) ? tolower(c) : c;
          if (++i >= 4) {
              while ((c = *p++) != ' ' && c != '\t') {
                  if (c == '>') break;
              }
              if (c == ' ' || c == '\t') {
                  while ( (c = *p++) == ' '  || c == '\t');
              }
              break;
          }
      }
      if (c != '>') {
          *q++ = (isupper(c)) ? tolower(c) : c;
          cnt = 0;
          while ( (c = *p++) != '>') {
              *q = (isupper(c)) ? tolower(c) : c;
              cnt++;
          }
          if (cnt) q++;

      }
  */
  while ((c = *p++) != '>') {
    if (c == ' ' || c == '\t' || c == '/' || c == '=')
      c = '_';
    else if (c >= 'a' && c <= 'z')
      c = toupper(c);
    *q++ = c;
  }
  *q = '\0';

  fprintf(outfile, "\n%s\n", "  /* "
                             "-------------------------------------------------"
                             "----------------- */");

  return;
}

static void
output()
{
  int i;

  current++;
  fprintf(outfile, "  /*\n");
  if (current > 1) {
    i = 0;
    while (savelin[i] == ' ' || savelin[i] == '\t')
      i++;
    fprintf(outfile, "   *\t%s ::= %s\n", name, &savelin[i]);
    fprintf(prodfile, "    \"%s ::= %s\",\n", name, &savelin[i]);
  } else {
    fprintf(outfile, "   *\t%s\n", savelin);
    fprintf(prodfile, "    \"%s\",\n", savelin);
  }
  fprintf(outfile, "   */\n");
  fprintf(outfile, "  case %s%d:\n", case_name, current);
  fprintf(outfile, "    break;\n");
}
