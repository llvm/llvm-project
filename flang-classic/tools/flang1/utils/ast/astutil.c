/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 *  \file
 *  \brief astutil.c - AST utility program
 */

#include "gbldefs.h"

#include "utils.h"
#ifndef _WIN64
#include <unistd.h>
#endif

#define ASTTMPFILE "ASTTMPFILE"

/*  define macros and data for input line types:  */
#define LT_SF 1
#define LT_ST 2
#define LT_SM 3
#define LT_SE 4
#define LT_SI 5
#define LT_FL 6

#define LT_TA 7

static LT elt[] = {{".SF", LT_SF}, {".ST", LT_ST}, {".SM", LT_SM},
                   {".SE", LT_SE}, {".SI", LT_SI}, {".FL", LT_FL},
                   {".TA", LT_TA}, {"", LT_EOF}};
static int lt; /* type of last line read */
static char tok[132];

struct symfld {
  char name[32]; /* field name */
  int size;      /* size in bytes */
  int offs;      /* offset in bytes starting from 0 ltor, ttob */
  int shared;    /* TRUE if shared */
  int flag;      /* TRUE if a flag */
};
#define MAXFIELDS 110
static struct symfld fields[MAXFIELDS];
static int fieldnum;

#define MAXATTRIB 20
static char attrnames[MAXATTRIB][32];
static int attrnum = 0;
static int addattr();
static void copy_file(FILE *, FILE *);

#define MAXATYPES 200
#define MAXFPERS 30
static struct symsym {
  char stype[32];       /* symbol type */
  char sname[32];       /* name for this sym type */
  int fields[MAXFPERS]; /* fields for this sym */
  int nfields;
  int attr[MAXATTRIB];
  int nattr;
} symbols[MAXATYPES];
static int symnum;

#define SYMLEN 19

static void proc_sym();
static void flushsym(int *, int, FILE *);
static void addfieldtosym(int *, int, int);
static void write_ast();
static int qscmp(const void *, const void *);

static int checkmode = 0;

/*************************************************************/

int
main(int aargc, char *aargv[])
{
  int argc;
  char **argv;
  argc = aargc;
  argv = aargv;
  if (argc > 1 && strcmp(argv[1], "-check") == 0) {
    checkmode = 1;
    argv[1] = argv[0];
    ++argv;
    --argc;
  }
  open_files(argc, argv);
  copy_file(IN3, OUT2); /* copy I_<name> macros created by symtab */
  proc_sym();           /* process symbol file */
  write_ast();          /* writeout ast macros */
  unlink(ASTTMPFILE);
  exit(0);
}

/**************************************************************/
static int findsym();
static int addfield(int sharedflag, int flagflag);

static void
proc_sym()
{
  int len;
  int i;
  int cursyms[20], cursym;
  FILE *outf;
  FILE *tempfp;
  int pdoffs;

  tempfp = NULL;
  outf = OUT1;
  lt = get_line1(IN1, elt, outf);
  fieldnum = 0;
  symnum = 0;
  cursym = 0;
  pdoffs = 0;
  LOOP
  { /* once for each line */
    switch (lt) {
    case LT_SF:
      /* shared field */
      get_token(tok, &len); /* field name */
      if (len <= 1)
        break;
      (void)addfield(TRUE, FALSE);
      break;

    case LT_ST:
      /* ast type (used to be symbol type) */
      get_token(tok, &len);
      if (len <= 1)
        break;
      (void)findsym();
      break;

    case LT_SM:
      /* ast type-specific fields */
      get_token(tok, &len);
      flushsym(cursyms, cursym, tempfp); /* flush info for current sym */
      cursym = 0;
      if (len == 1) {
        /* end of SM definitions */
        outf = OUT1;
        goto again;
      }
      while (len > 0) {
        cursyms[cursym++] = findsym();
        get_token(tok, &len);
      }
      /* open file to hold up to next SM */
      if ((tempfp = fopen(ASTTMPFILE, "wb+")) == NULL) {
        put_error(4, "Can't open temp file ASTTMPFILE");
      }
      outf = tempfp;
      flush_line(OUT1);
      goto again;

    case LT_SI:
      get_token(tok, &len);
      i = cursyms[0];
      strncpy(symbols[i].sname, tok, 31);
      symbols[i].sname[31] = 0;

      get_token(tok, &len);
      while (len > 0) {
        int jj;
        jj = addattr();
        symbols[i].attr[symbols[i].nattr++] = jj;
        get_token(tok, &len);
      }
      break;

    case LT_SE:
      get_token(tok, &len);
      i = addfield(FALSE, FALSE);
      addfieldtosym(cursyms, cursym, i);
      break;

    case LT_FL:
      get_token(tok, &len);
      i = addfield(FALSE, TRUE);
      addfieldtosym(cursyms, cursym, i);
      break;

    case LT_EOF:
      goto done;
    default:
      put_err(4, "Unknown LT: can't happen\n");
    }
    flush_line(outf);
  again:
    lt = get_line1(IN1, elt, outf);
  }
done:;
}

static void
write_ast()
{
  int fptrs[MAXFIELDS];
  int i, j;
  int k;
  char buff[32];
  char buff1[32];

  for (i = 0; i < fieldnum; ++i)
    fptrs[i] = i;
  qsort((char *)fptrs, fieldnum, sizeof(int), qscmp);
  /* read the ast.h boilerplate file and write stuff */
  lt = get_line1(IN2, elt, OUT2);
  LOOP
  {
    switch (lt) {
    case LT_ST: /* print symbol types */
      for (i = 0; i < symnum; ++i)
        fprintf(OUT2, "#define A_%s %d\n", symbols[i].stype, i);
      fprintf(OUT2, "#define AST_MAX %d\n", symnum - 1);
      fprintf(OUT2, "\n");
      break;
    case LT_SE: /* print fields access macros */
      for (i = 0; i < fieldnum; ++i) {
        j = fptrs[i];
        if (fields[j].flag)
          sprintf(buff, "f%d", fields[j].offs);
        else if (fields[j].shared) {
          if (strcmp("flags", fields[j].name) == 0)
            continue;
          strcpy(buff1, fields[j].name);
          for (k = 0; buff1[k] != '\0'; ++k)
            if (isupper(buff1[k]))
              buff1[k] = tolower(buff1[k]);
          sprintf(buff, "%s", buff1);
        } else {
          switch (fields[j].size) {
          case 1:
            put_error(2, "byte field not supported");
            sprintf(buff, "b%d", fields[j].offs + 1);
            break;
          case 2:
            sprintf(buff, "hw%d", fields[j].offs / 2 + 1);
            break;
          case 4:
            sprintf(buff, "w%d", fields[j].offs / 4 + 1);
            break;
          default:
            put_err(2, "Field not b,h,w in macro");
            strcpy(buff, "w0");
            break;
          }
        }
        if (!checkmode || fields[j].shared) {
          fprintf(OUT2, "#define A_%sG(s)   astb.stg_base[s].%s\n", fields[j].name,
                  buff);
          fprintf(OUT2, "#define A_%sP(s,v) (astb.stg_base[s].%s = (v))\n",
                  fields[j].name, buff);
        } else {
          /* output code to check that the type is ok for this access */
          fprintf(OUT2, "#define A_%sG(s)   "
                        "(astb.stg_base[(ast_type_check[astb.stg_base[s].type][%d]?("
                        "interr(\"bad A_%sG access, "
                        "A_TYPE=\",astb.stg_base[s].type,3),0):0),(s)].%s)\n",
                  fields[j].name, i, fields[j].name, buff);
          fprintf(OUT2, "#define A_%sP(s,v)   "
                        "(astb.stg_base[(ast_type_check[astb.stg_base[s].type][%d]?("
                        "interr(\"bad A_%sP access, "
                        "A_TYPE=\",astb.stg_base[s].type,3),0):0),(s)].%s = (v))\n",
                  fields[j].name, i, fields[j].name, buff);
        }
      }
      if (checkmode) {
        fprintf(OUT2, "extern char ast_type_check[AST_MAX+1][%d];\n", fieldnum);
      }
      break;
    case LT_TA: /* print type attributes */
      for (i = 0; i < attrnum; ++i)
        fprintf(OUT2, "#define __A_%s %d\n", attrnames[i], 1 << i);
      fprintf(OUT2, "\n");
      break;
    case LT_EOF:
      goto done;
    default:
      put_error(2, "Unknown line type");
      break;
    }
    lt = get_line1(IN2, elt, OUT2);
  }
done:;
  copy_file(IN4, OUT3); /* copy intast_sym inits created by symtab */
  /* write dinit for ASTB */
  fprintf(OUT3, "\n#pragma GCC diagnostic push\n");
  fprintf(OUT3, "#pragma GCC diagnostic ignored \"-Wmissing-field-initializers\"\n");
  fprintf(OUT3, "ASTB astb = {\n");
  fprintf(OUT3, "    {");
  /* char           *atypes[AST_MAX + 1]; */
  j = 6;
  for (i = 0; i < symnum; ++i) {
    if ((j += (k = strlen(symbols[i].sname) + 3)) > 80) {
      fprintf(OUT3, "\n     ");
      j = 6 + k;
    }
    fprintf(OUT3, "\"%s\",", symbols[i].sname);
  }
  fprintf(OUT3, "},\n");

  fprintf(OUT3, "    {0,\n");
  for (i = 1; i < symnum; ++i) {
    k = symbols[i].nattr;
    if (k == 0)
      fprintf(OUT3, "     0,\n");
    else {
      fprintf(OUT3, "     __A_%s", attrnames[symbols[i].attr[0]]);
      for (j = 1; j < k; ++j)
        fprintf(OUT3, "|__A_%s", attrnames[symbols[i].attr[j]]);
      fprintf(OUT3, ",\n");
    }
  }
  fprintf(OUT3, "    },\n");
  fprintf(OUT3, "};\n");
  fprintf(OUT3, "#pragma GCC diagnostic pop\n");

  if (checkmode) {
    char ch;
    int s, i, j, sf, x;
    fprintf(OUT3, "\n\nchar ast_type_check[AST_MAX+1][%d] = {\n", fieldnum);
    for (s = 0; s < symnum; ++s) {
      ch = ' ';
      for (i = 0; i < fieldnum; ++i) {
        j = fptrs[i];
        x = 1;
        if (fields[j].shared) {
          x = 0;
        } else {
          for (sf = 0; sf < symbols[s].nfields; ++sf) {
            if (symbols[s].fields[sf] == j)
              break;
          }
          if (sf < symbols[s].nfields) {
            x = 0;
          }
        }
        fprintf(OUT3, "%c%d", ch, x);
        ch = ',';
      }
      if (s < symnum - 1) {
        fprintf(OUT3, ", /* A_%s */\n", symbols[s].stype);
      } else {
        fprintf(OUT3, "}; /* A_%s */\n", symbols[s].stype);
      }
    }
    for (i = 0; i < fieldnum; ++i) {
      j = fptrs[i];
      fprintf(OUT3, "/* field %2d = A_%sG */\n", i, fields[j].name);
    }
  }
}

/*
 * I made sure the sort is stable, so it will give the same
 * results regardless of the host machine.  It was returning
 * '0' (no order) for fields with the same offset, which different
 * qsort implementations would order differently, so the ast.h file
 * would get updated for no good reason; my solution was to
 * compare the field names in that case
 */
static int
qscmp(const void *a1, const void *a2)
{
  int r;
  const int *f1, *f2;
  f1 = (const int *)a1;
  f2 = (const int *)a2;
  if (fields[*f1].flag && fields[*f2].flag) {
    r = fields[*f1].offs - fields[*f2].offs;
    if (r == 0)
      r = strcmp(fields[*f1].name, fields[*f2].name);
  } else if (fields[*f1].flag)
    r = -1;
  else if (fields[*f2].flag)
    r = 1;
  else {
    r = fields[*f1].offs - fields[*f2].offs;
    if (r == 0)
      r = strcmp(fields[*f1].name, fields[*f2].name);
  }
  return r;
}

static int chk_overlap(int f1, int f2, int flag);

static void
flushsym(int *cursyms, int cursym, FILE *tempf)
{
  int i, j, k;
  int indir;
  int addit;
  int *p;
  int offs;
  int output;
  int last;

  if (cursym == 0 || tempf == NULL)
    return;
  /* add shared fields not already present */
  for (i = 0; i < cursym; ++i) {
    indir = cursyms[i];
    for (j = 0; j < fieldnum; ++j) {
      if (!fields[j].shared)
        continue;
      addit = TRUE;
      for (k = 0; k < symbols[indir].nfields; ++k)
        if (chk_overlap(j, symbols[indir].fields[k], FALSE)) {
          addit = FALSE;
          break;
        }
      if (addit)
        symbols[indir].fields[symbols[indir].nfields++] = j;
    }
    qsort((char *)symbols[indir].fields, symbols[indir].nfields, sizeof(int),
          (int (*)())qscmp);
  }

  /* write symbol picture */
  indir = cursyms[0];
  fputs(".sz -4\n", OUT1);
  fputs(".TS\n", OUT1);
  fputs("tab(%);\n", OUT1);
  j = 0;
  p = symbols[indir].fields;
  k = symbols[indir].nfields;
  while (j < k && fields[*p].flag) {
    ++j;
    ++p;
  }
  offs = 0;
  last = 1;
  /* write the table format lines */
  for (i = 0; i < SYMLEN; ++i) {
    fputs("n cw(1.0i) sw(1.0i) sw(1.0i) sw(1.0i)\n", OUT1);
    fputs("n ", OUT1);
    output = 0;
    last = 1;
    while (output < 4) {
      if (j < k && fields[*p].offs == offs) {
        switch (fields[*p].size) {
        case 1:
          fputs("| cw(1.0i) ", OUT1);
          break;
        case 2:
          fputs("| cw(1.0i) sw(1.0i) ", OUT1);
          break;
        case 3:
          fputs("| cw(1.0i) sw(1.0i) sw(1.0i) ", OUT1);
          break;
        case 4:
          fputs("| cw(1.0i) sw(1.0i) sw(1.0i) sw(1.0i) ", OUT1);
          break;
        default:
          put_err(4, "Bad size in field");
          return;
        }
        last = 1;
        offs += fields[*p].size;
        output += fields[*p].size;
        ++j;
        ++p;
      } else {
        if (last)
          fputs("| ", OUT1);
        fputs("cw(1.0i) ", OUT1);
        ++output;
        ++offs;
        last = 0;
      }
    }
    fputs("|\n", OUT1);
  }
  fputs("n cw(1.0i) sw(1.0i) sw(1.0i) sw(1.0i) .\n", OUT1);

  j = 0;
  p = symbols[indir].fields;
  k = symbols[indir].nfields;
  while (j < k && fields[*p].flag) {
    ++j;
    ++p;
  }
  offs = 0;
  /* write the data lines */
  for (i = 0; i < SYMLEN; ++i) {
    fputs("%_\n", OUT1);
    fprintf(OUT1, "%d", i + 1);
    output = 0;
    while (output < 4) {
      if (j < k && fields[*p].offs == offs) {
        fprintf(OUT1, "%%%s", fields[*p].name);
        offs += fields[*p].size;
        output += fields[*p].size;
        ++j;
        ++p;
      } else {
        fputc('%', OUT1);
        ++output;
        ++offs;
      }
    }
    fputc('\n', OUT1);
  }
  fputs("%_\n", OUT1);
  fputs(".TE\n", OUT1);
  fputs(".sz +4\n", OUT1);

  /* append temp file contents to troff output */
  rewind(tempf);
  {
    char buffer[133];
    buffer[132] = 0;
    while (fgets(buffer, 132, tempf))
      fputs(buffer, OUT1);
  }

  /* close temp file */
  fclose(tempf);
}

static void
addfieldtosym(int *cursyms, int cursym, int field)
{
  int i, indir, j, k;

  for (i = 0; i < cursym; ++i) {
    indir = cursyms[i]; /* symbol number */
    for (j = 0; j < symbols[indir].nfields; ++j) {
      if (field == (k = symbols[indir].fields[j])) {
        put_err(2, "Field already specified for this sym");
        goto again;
      }
      if (chk_overlap(field, k, TRUE))
        goto again;
    }
    symbols[indir].fields[symbols[indir].nfields++] = field;
  again:;
  }
}

static int
chk_overlap(int f1, int f2, int flag)
{
  if (fields[f1].flag && fields[f2].flag) {
    if (fields[f1].offs == fields[f2].offs) {
      if (flag)
        put_err(2, "Flag overlaps one already defined");
      return TRUE;
    }
  } else if (fields[f1].flag || fields[f2].flag)
    return FALSE;
  /* check for field overlap */
  if (fields[f1].offs + fields[f1].size <= fields[f2].offs ||
      fields[f2].offs + fields[f2].size <= fields[f1].offs)
    return FALSE; /* they're disjoint */
  if (flag)
    put_err(2, "Field overlaps one already defined");
  return TRUE;
}

static int
findsym()
{
  int i;

  for (i = 0; i < symnum; ++i)
    if (strcmp(tok, symbols[i].stype) == 0)
      return i;
  if (symnum >= MAXATYPES) {
    put_error(2, "Too many symbol types");
    return 0;
  }
  strncpy(symbols[symnum].stype, tok, 31);
  symbols[symnum].stype[31] = 0;
  return symnum++;
}

static int
addfield(int sharedflag, int flagflag)
{
  int i;
  int size, offs;
  int len;
  char *aftp; /* position after w<d> */

  for (i = 0; i < fieldnum; ++i)
    if (strcmp(tok, fields[i].name) == 0)
      return i;

  /* add it */
  if (fieldnum >= MAXFIELDS) {
    put_error(2, "Too many symbol fields");
    return 0;
  }
  strncpy(fields[fieldnum].name, tok, 31);
  fields[fieldnum].name[31] = 0;
  get_token(tok, &len);
  if (len <= 0) {
    put_error(2, "Field location not specified");
    goto fixup;
  }
  /* parse location */
  if (flagflag) {
    offs = 0;
    size = 0;
    if (*tok != 'f')
      goto badloc;
    sscanf(tok + 1, "%d", &offs);
    if (offs == 0)
      goto badloc;
    goto done;
  }
  if (*tok != 'w' || tok[1] < '1')
    goto badloc;
  offs = 0;
  sscanf(tok + 1, "%d", &offs);
  if (offs == 0 || offs > SYMLEN)
    goto badloc;
  aftp = tok + 2;
  if (offs > 9)
    aftp++;
  offs = (offs - 1) * 4;
  if (aftp[0] == 0) {
    size = 4;
  } else if (aftp[0] != ':') {
    put_error(2, ": must follow word spec");
    size = 4;
  } else if (aftp[1] != 'h' && aftp[1] != 'b') {
    put_error(2, "Bad subfield spec");
    size = 4;
  } else if (aftp[1] == 'h') {
    size = 2;
    if (aftp[2] < '1' || aftp[2] > '2')
      put_error(2, "Bad halfword spec");
    else
      offs += (aftp[2] - '1') * 2;
  } else if (aftp[1] == 'b') {
    size = 1;
    if (aftp[2] < '1' || aftp[2] > '4')
      put_error(2, "Bad byte spec");
    else
      offs += aftp[2] - '1';
    if (aftp[3] == '-') {
      if (strcmp(fields[fieldnum].name, "flags") != 0 || aftp[4] < '1' ||
          aftp[4] > '4')
        put_error(2, "Bad flags spec");
      else
        size = (aftp[4] - aftp[2]) + 1;
    }
  } else {
    size = 0;
  }
done:
  fields[fieldnum].size = size;
  fields[fieldnum].offs = offs;
  fields[fieldnum].shared = sharedflag;
  fields[fieldnum].flag = flagflag;
  return fieldnum++;

badloc:
  put_error(2, "Bad field location");
/**** fall thru ****/
fixup:
  fields[fieldnum].size = fields[fieldnum].offs = 0;
  fields[fieldnum].shared = FALSE;
  fields[fieldnum].flag = FALSE;
  return fieldnum++;
}

/**************************************************************/

static int
addattr()
{
  int i;

  for (i = 0; i < attrnum; ++i)
    if (strcmp(tok, attrnames[i]) == 0)
      return i;
  if (attrnum >= MAXATTRIB) {
    put_error(2, "Too many type attributes");
    return 0;
  }
  strncpy(attrnames[attrnum], tok, 31);
  attrnames[attrnum][31] = 0;
  return attrnum++;
}

static void
copy_file(FILE *from, FILE *to)
{
  int c;
  fprintf(to, "\n/*----- begin symtab contribution -----*/\n");
  while ((c = fgetc(from))) {
    if (c == EOF)
      break;
    fputc(c, to);
  }
  fprintf(to, "/*----- end symtab contribution -----*/\n\n");
}
