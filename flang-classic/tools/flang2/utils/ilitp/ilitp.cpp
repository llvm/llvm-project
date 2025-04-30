/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*  ilitp.cpp - for ILI generation.

    utility program which reads ili definition file, ilitp.n,
    and writes file of macro and data definitions:
        iliatt.h
        ilinfodf.h

    nroff include directives of the form ".so filename" are recognized.
    The filename is interpreted relative to the including file.

    Lines beginning with '.' and containing one or more sets of the form
        {x,y,z}
    are expanded into multiple lines, each with one of the choices.
    If more than one set appears on the line, then choices are matched by
    position.

    E.g., a line ".IL {x,y,z} foo {X,Y,Z}" expands to three lines:
        .IL x foo X
        .IL y foo Y
        .IL z foo Z
    An error will be reported if the sets on a line have differing
    number of elements.  Furthermore, subsequent .AT, .CG, and .SI
    lines with {...} will be matched up with the .IL lines.

    The program assumes a non-adversarial user.  Some possible buffer
    overruns are unchecked.
*/

#include "universal.h"
#include "scutil.h"
#if defined(BIGOBJ)
#define ISZ_T BIGINT
#define UISZ_T BIGUINT
#else
#define ISZ_T INT
#define UISZ_T UINT
#endif
#include "sharedefs.h"
#include "ili.h"
#include "utils.h"

static int opcode_num = 0;
static int max_ili_num;
static std::string buff;

// Flags for tracking which part of an instruction description we've seen so far.
struct processed_flags {
  bool il; //!< .IL line (or similar) has been seen.
  bool cg; //!< .CG line has been seen.
  bool at; //!< .AT line has been seen  
};

// Indexed by opcode.  A vector is needed because "ILI template expansion"
// delivers the expansion of a line as a group of consecutive lines.
std::vector<processed_flags> processed;

/*static*/ ILIINFO ilis[1050]; /* declared external in ili.h  */
SCHINFO schinfo[1050];

static void do_IL_line(int pass);
static void do_AT_line(void);
static void do_CG_line(void);
static void do_latency(const char *lat, int shift);
static void do_SI_line(void);
static int lookup_ili(const char *name);
static void error(const char *text, const char *additional_text=nullptr);

#define STASH(string) strcpy((char*)malloc(strlen(string) + 1), string)

//! Return true iff s begins with first 3 characters of nroffCmd.
static bool
match(const std::string& s, const char *nroffCmd) {
  return strncmp(s.c_str(), nroffCmd, 3) == 0;
}

/* True if text in buff introduces an ILI instruction;
   e.g. is a ".IL" line. */
static bool
introduces_ILI(const std::string& buff)
{
  if (match(buff, ".IL"))
    return true;
#if defined(PGFTN) || defined(PGF90)
  if (match(buff, ".FL"))
    return true;
#elif defined(PGOCL)
  if (match(buff, ".OL"))
    return true;
  if (match(buff, ".CL"))
    return true;
#else
  if (match(buff, ".CL"))
    return true;
#endif
  return false;
}

/* True if text in buff elborates on a previously introduced
   n ILI instruction. */
static bool
elaborates_ILI(const std::string& buff)
{
  if (match(buff, ".AT"))
    return true;
  if (match(buff, ".CG"))
    return true;
  if (match(buff, ".SI"))
    return true;
  return false;
}

/* Text containing {...} that needs to be expanded. */
static std::string template_buff;

/* Number of choices in template that introduced an ILI instruction.
   Related lines such as .AT inherit this value from the line
   that introduced the instruction.  -1 if not processing a template. */
static int template_size;

/* -1 if not expanding, otherwise 1-based index of current choice to expand
   (or has been expanded) in a {...}.  A value of 0 is used transiently
   after a template line is first scanned and before the first choice
   is selected. */
static int template_index = -1;

/* Value of opcode_num just before a "introduces ILI" line was scanned. */
static int template_base_opcode_num = -1;

static int
get_template_size(std::string& buff)
{
  /* Caller guarantees that '{' exists in buff. */
  const char *s = strchr(buff.c_str(), '{');
  int size = 1;
  while (*++s != '}') {
    if (*s == ',')
      ++size;
    if (!*s)
      error("{... without closing }");
  }
  return size;
}

/* Get expansion of template_buff, as determined by template_index.
   The index  is one-based.

   For example, if a line has:
       .IL {x,y,z} foo {X,Y,Z}
   and template_index==2, then buff will be filled with:
       .IL y foo Y
*/
static void
get_template_expansion(std::string& buff)
{
  const char *p = template_buff.c_str();
  buff.clear();
  for (;;) {
    int i;
    /* Copy regular characters */
    for (; *p != '{'; ++p) {
      buff.push_back(*p);
      if (!*p)
        return;
    }
    ++p; /* Skip brace */
    /* Find choice indexed by template_index. */
    for (i = 1; i != template_index; ++i) {
      /* Look for comma */
      for (; *p != ','; ++p) {
        if (!*p)
          error("{... without closing '}'");
        if (*p == '}')
          error("{...} with fewer choices than expected");
      }
      ++p; /* skip comma */
    }
    /* Copy the choice. */
    while (*p != ',' && *p != '}')
      buff.push_back(*p++);
    /* Check for user error. */
    if (template_index == template_size) {
      /* Should be last choice. */
      if (*p != '}')
        error("{...} with more choices than expected");
    }
    /* Advance to closing brace and skip it. */
    p = strchr(p, '}');
    if (!p)
      error("{... without closing brace");
    ++p;
  }
}

static NroffInStream input_file;

/** Open an input file and push it on the file filestack. */
static void
open_input_file(const char *filename)
{
  input_file.open(filename);
  if (!input_file)
    error("unable to open");
}

/** Get next line of input, expanding any include files (.so directives)
    or implicit templates ({...}). */
static bool
get_input_line(std::string& buff)
{
  for(;;) {
    if (template_index >= 0) {
      /* Expanding a template */
      ++template_index;
      if (template_index > template_size) {
        /* Done expanding the template.  Do not reset template_size
           or template_base_opcode_num, because these values are inherited
           by subsequent .AT lines. */
        template_index = -1;
      } else {
        if (!introduces_ILI(template_buff))
          /* Set opcode_num to correspond with earlier .IL etc. line. */
          opcode_num = template_base_opcode_num + template_index;
        get_template_expansion(buff);
        return true;
      }
    }
    if (!getline(input_file, buff)) 
      return false;
    if (introduces_ILI(buff)) {
      if (buff.find('{') != std::string::npos) {
        /* Line will expand into multiple lines.
           Remember the value of opcode_num so we can reset it later
           to the right value when expanding the lines that correspond
           to these. */
        template_buff = buff;
        template_index = 0;
        template_size = get_template_size(buff);
        template_base_opcode_num = opcode_num;
        continue;
      } else {
        /* Do not not expand the line or elaborations of the same ILI. */
        template_size = -1;
      }
    } else if (elaborates_ILI(buff)) {
      if (template_size >= 0) {
        /* Need to expand line analogously to line that introduced this ILI. 
           Line inherits template_size and template_base_opcode_num from 
           earlier line. */
        template_buff = buff;
        template_index = 0;
        continue;
      }
    }
    /* regular line */
    return true;
  }
}

/* ------------------------------------------------------- */

int
main(int argc, char *argv[])
{
  int i;
  int inputFileIndex = 0;
  FILE *fdiliatt, *fdilinfo, *fdschinfo;

  /*  ---- step 0: collect commandline -I<include file paths>:   */
  
  collectIncludePaths(argc, argv);

  /*  ---- step 1: open input and output files:   */

  for (i = 1; i < argc; i++ ) {
    if( strncmp(argv[i], "-I", 2) == 0 ) { 
      continue;
    }
    if (inputFileIndex != 0) {
      error("More than one input file specified\n");
    } else {
      inputFileIndex = i;
    }
  }

  const char *input_file_name = inputFileIndex != 0? argv[inputFileIndex] : "ilitp.n";
  open_input_file(input_file_name);

  fdiliatt = fopen("iliatt.h", "wb");
  if (fdiliatt == NULL)
    error("unable to open output file iliatt.h");
  fdilinfo = fopen("ilinfodf.h", "wb");
  if (fdilinfo == NULL)
    error("unable to open output file ilinfodf.h");
  fdschinfo = fopen("schinfo.h", "wb");
  if (fdschinfo == NULL)
    error("unable to open output file schinfo.h");

  /*  ---- step 2: scan input file to read in ili names from IL lines: */

  while (get_input_line(buff))
    if (introduces_ILI(buff))
      do_IL_line(1);

  max_ili_num = opcode_num;
  processed.resize(max_ili_num+1);
  opcode_num = 0;

  /*  ---- step 3: rescan input file to process all relevant lines:  */

  open_input_file(input_file_name);
  while (get_input_line(buff)) {
    if (introduces_ILI(buff))
      do_IL_line(2);
    if (match(buff, ".AT"))
      do_AT_line();
    if (match(buff, ".CG"))
      do_CG_line();
    if (match(buff, ".SI"))
      do_SI_line();
  }

  /*  -------------- step 4:  write output file - iliatt.h:  */

  fprintf(fdiliatt,
          "/* iliatt.h - DO NOT EDIT!\n"
          "   This file was automatically generated by program %s. */\n\n",
          argv[0]);
  /* Write as an enum ILI_OP */
  fprintf(fdiliatt, "typedef enum ILI_OP {\n");
  fprintf(fdiliatt, "    IL_%-24.24s = %d,\n", "NONE", 0);
  for (i = 1; i <= opcode_num; i++)
    fprintf(fdiliatt, "    IL_%-24.24s = %d,\n", ilis[i].name, i);
  fprintf(fdiliatt, "    %-24.24s    = %d\n", "GARB_COLLECTED", 0xffff);
  fprintf(fdiliatt, "} ILI_OP;\n\n");
  /* Write as reflexive define */
  for (i = 1; i <= opcode_num; i++)
    fprintf(fdiliatt, "#define IL_%-24.24s IL_%s\n", ilis[i].name,
            ilis[i].name);
  fprintf(fdiliatt, "#define N_ILI %d\n", opcode_num + 1);

  fclose(fdiliatt);

  /*  -------------- step 5:  write output file - ilinfodf.h:  */

  fprintf(fdilinfo,
          "/* ilinfodf.h - DO NOT EDIT!\n"
          "   This file was automatically generated by program %s. */\n",
          argv[0]);
  fprintf(fdilinfo, "\nILIINFO ilis[%d] = {\n", opcode_num + 1);

  /* Make sure to update the DUMMY entry if fields are added to ILIINFO.
   */
  fprintf(fdilinfo,
          " { \"DUMMY\",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'X',"
          "{0, 0, 0, 0} },\n");

  for (i = 1; i <= opcode_num; i++) {
    int j;

    fprintf(fdilinfo, " { \"%s\", ", ilis[i].name);

    if (ilis[i].opcod != 0)
      fprintf(fdilinfo, "%s, %d, 0x%04x, %d, %d, ", ilis[i].opcod, ilis[i].oprs,
              ilis[i].attr, ilis[i].notCG, ilis[i].CGonly);
    else /* no string specified for opcod: */
      fprintf(fdilinfo, "0, %d, 0x%04x, %d, %d, ", ilis[i].oprs, ilis[i].attr,
              ilis[i].notCG, ilis[i].CGonly);

    fprintf(
        fdilinfo,
        "\n\t  %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,",
        ilis[i].notAILI, ilis[i].terminal, ilis[i].move, ilis[i].memdest,
        ilis[i].ccarith, ilis[i].cclogical, ilis[i].ccmod, ilis[i].shiftop,
        ilis[i].memarg, ilis[i].ssedp, ilis[i].ssest,
        ilis[i].conditional_branch, ilis[i].sse_avx, ilis[i].avx_only,
        ilis[i].avx_special, ilis[i].avx3_special, ilis[i].asm_special,
        ilis[i].asm_nop, ilis[i].accel, ilis[i].replaceby);

    if (ilis[i].size == 0)
      fprintf(fdilinfo, " '\\0', {");
    else
      fprintf(fdilinfo, " '%c', {", ilis[i].size);

    for (j = 0; j < MAX_OPNDS - 1; j++)
      fprintf(fdilinfo, " %d,", ilis[i].oprflag[j]);

    fprintf(fdilinfo, " %d} }, \t/* %d */\n", ilis[i].oprflag[j], i);
  }

  fprintf(fdilinfo, "};\n");
  fclose(fdilinfo);

  /*  -------------- step 6:  write output file - schinfo.h:  */

  fprintf(fdschinfo,
          "/* schinfo.h - DO NOT EDIT!\n"
          "   This file was automatically generated by program %s. */\n",
          argv[0]);
  fprintf(fdschinfo, "\nSCHINFO schinfo[%d] = {\n", opcode_num + 1);

  fprintf(fdschinfo, "{0,0},\t/* DUMMY */\n");
  for (i = 1; i <= opcode_num; i++) {
    fprintf(fdschinfo, " {");
    fprintf(fdschinfo, "%#10.8x, ", schinfo[i].latency);
    fprintf(fdschinfo, "%#10.8x", schinfo[i].attrs);
    fprintf(fdschinfo, "},\t/* %s */\n", ilis[i].name);
  }

  fprintf(fdschinfo, "};\n");
  fclose(fdschinfo);

  exit(0);
} /* end main(int argc, char *argv[]) */

/* -------------------------------------------------------- */

static void
do_IL_line(int pass /* 1 or 2 */)
/*
 * Process an opcode definition line (begins with ".IL" or ".FL").
 */
{
  int token_count;
  char name[100], token[10][100];

  opcode_num++;
  if (opcode_num >= (int)(sizeof(schinfo) / sizeof(SCHINFO))) {
    error("Number of ILI opcodes exceeds sizeof(schinfo)");
  }

  token_count = sscanf(buff.c_str(), ".%*cL %s %s %s %s %s %s %s %s %s %s %s", &name[0],
                       token[0], token[1], token[2], token[3], token[4],
                       token[5], token[6], token[7], token[8], token[9]);

  if (pass == 1) {
    static ILIINFO zero;

    if (token_count >= 11)
      error("too many operands on ili line");
    if (token_count > MAX_OPNDS + 1)
      error("maximum number of ili operands exceeded");
    if (token_count < 1)
      error("ili name missing from .IL line");
    ilis[opcode_num] = zero;
    ilis[opcode_num].name = STASH(name);
  } else { /* pass 2 */
    char *ofp;
    int i, k;

    if (processed[opcode_num].il)
      error("internal error -- duplicate opcode_num");
    processed[opcode_num].il = true;

    ofp = ilis[opcode_num].oprflag;
    for (i = 0; i < MAX_OPNDS; i++)
      ofp[i] = 0;
    for (i = 0; i < token_count - 1; i++) {
      k = 0;
      if (strcmp(token[i], "sym") == 0)
        k = ILIO_SYM;
      if (strcmp(token[i], "stc") == 0)
        k = ILIO_STC;
      if (strcmp(token[i], "nme") == 0)
        k = ILIO_NME;
      if (strcmp(token[i], "ir") == 0)
        k = ILIO_IR;
      if (strcmp(token[i], "kr") == 0)
        k = ILIO_KR;
      if (strcmp(token[i], "hp") == 0)
        k = ILIO_HP;
      if (strcmp(token[i], "sp") == 0)
        k = ILIO_SP;
      if (strcmp(token[i], "dp") == 0)
        k = ILIO_DP;
      /* just for debug to dump ili */
      if (strcmp(token[i], "qp") == 0)
        k = ILIO_QP;
      if (strcmp(token[i], "cs") == 0)
        k = ILIO_CS;
      if (strcmp(token[i], "cd") == 0)
        k = ILIO_CD;
      if (strcmp(token[i], "ar") == 0)
        k = ILIO_AR;
      if (strcmp(token[i], "xmm") == 0)
        k = ILIO_XMM;
      if (strcmp(token[i], "x87") == 0)
        k = ILIO_X87;
      if (strcmp(token[i], "doubledouble") == 0)
        k = ILIO_DOUBLEDOUBLE;
      if (strcmp(token[i], "lnk") == 0)
        k = ILIO_LNK;
      if (strcmp(token[i], "irlnk") == 0)
        k = ILIO_IRLNK;
      if (strcmp(token[i], "hplnk") == 0)
        k = ILIO_HPLNK;
      if (strcmp(token[i], "splnk") == 0)
        k = ILIO_SPLNK;
      if (strcmp(token[i], "dplnk") == 0)
        k = ILIO_DPLNK;
      if (strcmp(token[i], "arlnk") == 0)
        k = ILIO_ARLNK;
      if (strcmp(token[i], "krlnk") == 0)
        k = ILIO_KRLNK;
      if (strcmp(token[i], "qplnk") == 0)
        k = ILIO_QPLNK;
      if (strcmp(token[i], "cslnk") == 0)
        k = ILIO_CSLNK;
      if (strcmp(token[i], "cdlnk") == 0)
        k = ILIO_CDLNK;
      if (strcmp(token[i], "cqlnk") == 0)
        k = ILIO_CQLNK;
      if (strcmp(token[i], "128lnk") == 0)
        k = ILIO_128LNK;
      if (strcmp(token[i], "256lnk") == 0)
        k = ILIO_256LNK;
      if (strcmp(token[i], "512lnk") == 0)
        k = ILIO_512LNK;
      if (strcmp(token[i], "x87lnk") == 0)
        k = ILIO_X87LNK;
      if (strcmp(token[i], "doubledoublelnk") == 0)
        k = ILIO_DOUBLEDOUBLELNK;
      if (strcmp(token[i], "float128lnk") == 0)
        k = ILIO_FLOAT128LNK;
      if (k == 0)
        error("unrecognized ili operand type");
      ofp[i] = k;
    }
    ilis[opcode_num].oprs = token_count - 1;
  }
} /* end do_IL_line(pass) */

/* -------------------------------------------------------- */

static void
do_AT_line(void)
/*
 * Process an attribute definition line (begins with ".AT").
 */
{
  static char attr[4][50], attr1[50], attr2[50], attr3[50];
  int token_count, i;
  int attrb = 0;

  if (!processed[opcode_num].il)
    error(".AT line should be preceded by .IL line");
  if (processed[opcode_num].at) 
    error("multiple .AT lines for the same opcode");
  processed[opcode_num].at = true;

  /* Parse .AT line.  First three tokens have fixed meaning, then
   * variable number of remaining attributes.
   */

  token_count = sscanf(buff.c_str(), ".AT %s %s %s %s %s %s %s", attr1, attr2, attr3,
                       attr[0], attr[1], attr[2], attr[3]);

  if (token_count < 3)
    error("at least 3 operands required on .AT line");

  if (token_count >= 7)
    error("too many attribute specs. on .AT line");

  /*  process first attribute - ili type:  */

  if (strcmp("arth", attr1) == 0)
    attrb = ILTY_ARTH;
  if (strcmp("branch", attr1) == 0)
    attrb = ILTY_BRANCH;
  if (strcmp("cons", attr1) == 0)
    attrb = ILTY_CONS;
  if (strcmp("define", attr1) == 0)
    attrb = ILTY_DEFINE;
  if (strcmp("load", attr1) == 0)
    attrb = ILTY_LOAD;
  if (strcmp("pload", attr1) == 0)
    attrb = ILTY_PLOAD;
  if (strcmp("move", attr1) == 0)
    attrb = ILTY_MOVE;
  if (strcmp("other", attr1) == 0)
    attrb = ILTY_OTHER;
  if (strcmp("proc", attr1) == 0)
    attrb = ILTY_PROC;
  if (strcmp("store", attr1) == 0)
    attrb = ILTY_STORE;
  if (strcmp("pstore", attr1) == 0)
    attrb = ILTY_PSTORE;
  if (attrb == 0)
    error("first ili attribute, ili type, not recognized");

  /*  process second attribute - "null" or "comm":   */

  if (strcmp("comm", attr2) == 0)
    attrb |= (1 << 4);
  else if (strcmp("null", attr2) != 0)
    error("second attribute must be 'null' or 'comm'");

  /*  process third attribute - ili return value type:  */

  if (strcmp("lnk", attr3) == 0)
    attrb |= (ILIA_LNK << 5);
  else if (strcmp("ir", attr3) == 0)
    attrb |= (ILIA_IR << 5);
  else if (strcmp("hp", attr3) == 0)
    attrb |= (ILIA_HP << 5);
  else if (strcmp("sp", attr3) == 0)
    attrb |= (ILIA_SP << 5);
  else if (strcmp("dp", attr3) == 0)
    attrb |= (ILIA_DP << 5);
  else if (strcmp("trm", attr3) == 0)
    attrb |= (ILIA_TRM << 5);
  else if (strcmp("ar", attr3) == 0)
    attrb |= (ILIA_AR << 5);
  else if (strcmp("kr", attr3) == 0)
    attrb |= (ILIA_KR << 5);
  else if (strcmp("qp", attr3) == 0)
    attrb |= (ILIA_QP << 5);
  else if (strcmp("cs", attr3) == 0)
    attrb |= (ILIA_CS << 5);
  else if (strcmp("cd", attr3) == 0)
    attrb |= (ILIA_CD << 5);
  else if (strcmp("cq", attr3) == 0)
    attrb |= (ILIA_CQ << 5);
  else if (strcmp("128", attr3) == 0)
    attrb |= (ILIA_128 << 5);
  else if (strcmp("256", attr3) == 0)
    attrb |= (ILIA_256 << 5);
  else if (strcmp("512", attr3) == 0)
    attrb |= (ILIA_512 << 5);
  else if (strcmp("x87", attr3) == 0)
    attrb |= (ILIA_X87 << 5);
  else if (strcmp("doubledouble", attr3) == 0)
    attrb |= (ILIA_DOUBLEDOUBLE << 5);
  else if (strcmp("float128", attr3) == 0)
    attrb |= (ILIA_FLOAT128 << 5);
  else
    error("unrecognized 3rd attribute of .AT line");

  /*  process remaining attributes:  */
  bool fence = false;
  bool atomicrmw = false;
  bool cmpxchg = false;
  for (i = 0; i < token_count - 3; i++) {
    if (strcmp(attr[i], "cse") == 0)
      attrb |= (ILIA_CSE << 10);
    else if (strcmp(attr[i], "dom") == 0)
      attrb |= (ILIA_DOM << 10);
    else if (strcmp(attr[i], "ssenme") == 0)
      attrb |= (1 << 12);
    else if (strcmp(attr[i], "vect") == 0)
      attrb |= (1 << 13);
    else if (strcmp(attr[i], "fence") == 0)
      fence = true;
    else if (strcmp(attr[i], "atomicrmw") == 0)
      atomicrmw = true;
    else if (strcmp(attr[i], "cmpxchg") == 0)
      cmpxchg = true;
    else
      error("unrecognized attribute on .AT line: ",attr[i]);
  }
  if (atomicrmw && cmpxchg)
    error("can have at most one of atomicrmw or atomic_cmpxchg attributes");
  if ((atomicrmw || cmpxchg) && !fence)
    error("fence attribute required if atomicrmw or atomic_cmpxchg attribute is present");
  attrb |= (cmpxchg ? 3 : atomicrmw ? 2 : fence ? 1 : 0) << 14;

  ilis[opcode_num].attr = attrb;

  /*  check that not both "cse" and "dom" specified:  */
  if (IL_IATYPE(opcode_num) == 3)
    error("conflicting attributes: dom and cse");
  if (IL_SSENME(opcode_num) && IL_OPRFLAG(opcode_num, 1) != ILIO_ARLNK)
    error("when ssenme attribute, 1st operand must be arlnk");
  if (IL_SSENME(opcode_num) && IL_OPRFLAG(opcode_num, 3) != ILIO_NME)
    error("when ssenme attribute, 3rd operand must be nme");

} /* end do_AT_line(void) */

/* -------------------------------------------------------- */

static void
do_CG_line(void)
/*
 * Process a ".CG" line which defines attributes for the Code Generator.
 */
{
  static char attr[10][50];
  int token_count, i;

  if (!processed[opcode_num].il)
    error(".CG line should be preceded by .IL line");
  if (processed[opcode_num].cg) 
    error("multiple .CG lines for the same opcode");
  processed[opcode_num].cg = true;

  /* Parse the ".CG" line.  0 or more tokens are allowed.
   */
  token_count = sscanf(buff.c_str(), ".CG %s %s %s %s %s %s %s %s %s %s", attr[0],
                       attr[1], attr[2], attr[3], attr[4], attr[5], attr[6],
                       attr[7], attr[8], attr[9]);

  if (token_count >= 10)
    error("too many attributes on .CG line");

  for (i = 0; i < token_count; i++) {
    if (strcmp(attr[i], "notCG") == 0) {
      ilis[opcode_num].notCG = 1;
    } else if (strcmp(attr[i], "CGonly") == 0) {
      ilis[opcode_num].CGonly = 1;
    } else if (strcmp(attr[i], "notAILI") == 0) {
      ilis[opcode_num].notAILI = 1;
    } else if (strcmp(attr[i], "replaceby") == 0) {
      i++;
      ilis[opcode_num].replaceby = lookup_ili(attr[i]);
    } else if (strcmp(attr[i], "terminal") == 0) {
      ilis[opcode_num].terminal = 1;
    } else if (strcmp(attr[i], "move") == 0) {
      ilis[opcode_num].move = 1;
    } else if (strcmp(attr[i], "memdest") == 0) {
      ilis[opcode_num].memdest = 1;
    } else if (strcmp(attr[i], "ccarith") == 0) {
      ilis[opcode_num].ccarith = 1, ilis[opcode_num].ccmod = 1;
    } else if (strcmp(attr[i], "cclogical") == 0) {
      ilis[opcode_num].cclogical = 1, ilis[opcode_num].ccmod = 1;
    } else if (strcmp(attr[i], "ccmod") == 0) {
      ilis[opcode_num].ccmod = 1;
    } else if (strcmp(attr[i], "shiftop") == 0) {
      ilis[opcode_num].shiftop = 1;
    } else if (strcmp(attr[i], "memarg") == 0) {
      ilis[opcode_num].memarg = 1;
    } else if (strcmp(attr[i], "ssedp") == 0) {
      ilis[opcode_num].ssedp = 1;
    } else if (strcmp(attr[i], "ssest") == 0) {
      ilis[opcode_num].ssest = 1;
    } else if (strcmp(attr[i], "conditional_branch") == 0) {
      ilis[opcode_num].conditional_branch = 1;
    } else if (strcmp(attr[i], "sse_avx") == 0) {
      ilis[opcode_num].sse_avx = 1;
    } else if (strcmp(attr[i], "avx_only") == 0) {
      ilis[opcode_num].avx_only = 1;
    } else if (strcmp(attr[i], "avx_special") == 0) {
      ilis[opcode_num].avx_special = 1;
    } else if (strcmp(attr[i], "avx3_special") == 0) {
      ilis[opcode_num].avx3_special = 1;
    } else if (strcmp(attr[i], "asm_special") == 0) {
      ilis[opcode_num].asm_special = 1;
    } else if (strcmp(attr[i], "asm_nop") == 0) {
      ilis[opcode_num].asm_nop = 1;
    } else if (strcmp(attr[i], "accel") == 0) {
      ilis[opcode_num].accel = 1;
    } else if (attr[i][0] == '"') {
      ilis[opcode_num].opcod = STASH(attr[i]);
    } else if (strcmp(attr[i], "'l'") == 0 || strcmp(attr[i], "'q'") == 0 ||
               strcmp(attr[i], "'w'") == 0 || strcmp(attr[i], "'b'") == 0 ||
               strcmp(attr[i], "'y'") == 0 || strcmp(attr[i], "'z'") == 0) {
      ilis[opcode_num].size = attr[i][1];
    } else
      error("unrecognized attribute on .CG line");
  }

  /* If 'replaceby' then also require 'notCG'.
   */

  if (ilis[opcode_num].replaceby && !ilis[opcode_num].notCG)
    printf("warning: notCG attribte required for ili %s.\n",
           ilis[opcode_num].name);

  /* If an AILI can have this opcode then either an instruction mnemonic
   * or 'asm_special' (or both) or 'asm_nop' must be specified.
   */

  if (!ilis[opcode_num].notCG && !ilis[opcode_num].notAILI) {
    if (!ilis[opcode_num].asm_special && !ilis[opcode_num].asm_nop &&
        ilis[opcode_num].opcod == NULL)
      printf("warning: ili %s is missing mnemonic spec.\n",
             ilis[opcode_num].name);
  }

  /* Set 'notAILI' if 'notCG' is true.
   */

  if (ilis[opcode_num].notCG)
    ilis[opcode_num].notAILI = 1;

} /* end do_CG_line(void) */

/* --------------------------------------------------------- */

/* parse latency attributes
 *     Format:
 *
 *         lat[R/R]
 *         lat[R/M:R/R]
 *
 * R/R: Reg/reg format, 8-bit unsigned integer value
 * R/M: Reg/Mem format, 8-bit unsigned integer value
 *
 * For the single value case (lat[R/R]), the same value is copied to
 * the R/M field for latency.
 *
 * The instructions that have different latencies for 8/16/32/64
 * operands (e.g. div, mul) are assumed to be encoded for their 64-bit
 * incarnations.  They are relatively few in number and can be handled
 * by special cases in the scheduler.  Similar assumption for
 * instructions with a latency range (e.g. 'int'), which are rare, and
 * not likely to be generated in any case.
 */
static void
do_latency(const char *lat, int shift)
{
  /* 'ld' and 'st' are flags */

  int rr_lat, rm_lat, lat_count;

  lat_count = sscanf(lat, "lat(%d:%d)", &rm_lat, &rr_lat);

  if (lat_count == 2) {
    if (shift)
      error("R/R and R/M latency are meaningless for loads/stores.");

    if (rr_lat > 255 || rm_lat > 255)
      error("Latency out of range on .SI line");
    schinfo[opcode_num].latency |=
        (((unsigned short)rm_lat << RM_SHIFT) | (unsigned short)rr_lat);
  } else {
    lat_count = sscanf(lat, "lat(%d)", &rr_lat);
    if (lat_count == 1) {
      if (rr_lat > 255)
        error("Latency out of range on .SI line");
      schinfo[opcode_num].latency |= ((unsigned short)rr_lat << shift);
    } else {
      error("Incorrect latency format on .SI line");
    }
  }
} /* end do_latency(char *lat, int shift) */

/* --------------------------------------------------------- */

static void
do_SI_line(void)
/*
 * Process a ".SI" line which defines latencies, etc.
 */
{
  static char attr[10][48];
  unsigned int attrs;
  int token_count, i;
  /* int ld_flg = 0, st_flg = 0, rm_flg = 0; */
  int shift;

  if (!processed[opcode_num].il)
    error(".SI line should be preceded by .IL line");

  token_count = sscanf(buff.c_str(), ".SI %s %s %s %s %s %s %s %s %s %s", attr[0],
                       attr[1], attr[2], attr[3], attr[4], attr[5], attr[6],
                       attr[7], attr[8], attr[9]);

  if (token_count >= 10) {
    error("too many attributes on .SI line");
  }

  if (strcmp(attr[0], "ld") == 0) {
    /* ld_flg = 1; */
    shift = LD_SHIFT;
  } else if (strcmp(attr[0], "st") == 0) {
    /* st_flg = 1; */
    shift = ST_SHIFT;
  } else if (strcmp(attr[0], "r/m") == 0) {
    /* rm_flg = 1; */
    shift = RM_SHIFT;
  } else {
    shift = 0;
  }

  for (i = 0; i < token_count; i++) {
    if (strcmp(attr[i], "fadd") == 0)
      schinfo[opcode_num].attrs |= P_FADD << shift;
    else if (strcmp(attr[i], "fmul") == 0)
      schinfo[opcode_num].attrs |= P_FMUL << shift;
    else if (strcmp(attr[i], "fst") == 0)
      schinfo[opcode_num].attrs |= P_FST << shift;
    else if (strcmp(attr[i], "direct") == 0)
      schinfo[opcode_num].attrs |= DEC_DIR << shift;
    else if (strcmp(attr[i], "double") == 0)
      schinfo[opcode_num].attrs |= DEC_DBL << shift;
    else if (strcmp(attr[i], "vector") == 0)
      schinfo[opcode_num].attrs |= DEC_VEC << shift;
    else if (strncmp(attr[i], "lat", 3) == 0)
      do_latency(attr[i], shift);
  }

  /* if (!rm_flg && !st_flg && !ld_flg) {
   *   schinfo[opcode_num].attrs |= (schinfo[opcode_num] & 0xff) << RM_SHIFT;
   * }
   */

  attrs = ((SCH_ATTR(opcode_num) >> shift) & (DEC_DIR | DEC_DBL | DEC_VEC));
  if (!(attrs == 0 || attrs == DEC_DIR || attrs == DEC_DBL ||
        attrs == DEC_VEC)) {
    error("Too many decode types listed on .SI line");
  }
} /* end do_SI_line(void) */

/* --------------------------------------------------------- */

static int
lookup_ili(const char *name)
{
  int i;

  for (i = 1; i <= max_ili_num; i++)
    if (strcmp(name, ilis[i].name) == 0)
      return i;

  error("reference to non-existent ili name");
  return 0;

} /* end lookup_ili(char *name) */

/* --------------------------------------------------------- */

/**
 * Print an error message.
 */
static void
error(const char *text, const char *additional_text)
{
  input_file.printError(FATAL, text, additional_text);
}

/* --------------------------------------------------------- */
