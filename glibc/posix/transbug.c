#include <stdio.h>
#include <ctype.h>
#include <errno.h>
#include <locale.h>
#include <regex.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

/* lowercase chars mapped to uppercase */
static const char casetable[] = {
  '\000', '\001', '\002', '\003', '\004', '\005', '\006', '\007',
  '\010', '\011', '\012', '\013', '\014', '\015', '\016', '\017',
  '\020', '\021', '\022', '\023', '\024', '\025', '\026', '\027',
  '\030', '\031', '\032', '\033', '\034', '\035', '\036', '\037',
  /* ' '     '!'     '"'     '#'     '$'     '%'     '&'     ''' */
  '\040', '\041', '\042', '\043', '\044', '\045', '\046', '\047',
  /* '('     ')'     '*'     '+'     ','     '-'     '.'     '/' */
  '\050', '\051', '\052', '\053', '\054', '\055', '\056', '\057',
  /* '0'     '1'     '2'     '3'     '4'     '5'     '6'     '7' */
  '\060', '\061', '\062', '\063', '\064', '\065', '\066', '\067',
  /* '8'     '9'     ':'     ';'     '<'     '='     '>'     '?' */
  '\070', '\071', '\072', '\073', '\074', '\075', '\076', '\077',
  /* '@'     'A'     'B'     'C'     'D'     'E'     'F'     'G' */
  '\100', '\141', '\142', '\143', '\144', '\145', '\146', '\147',
  /* 'H'     'I'     'J'     'K'     'L'     'M'     'N'     'O' */
  '\150', '\151', '\152', '\153', '\154', '\155', '\156', '\157',
  /* 'P'     'Q'     'R'     'S'     'T'     'U'     'V'     'W' */
  '\160', '\161', '\162', '\163', '\164', '\165', '\166', '\167',
  /* 'X'     'Y'     'Z'     '['     '\'     ']'     '^'     '_' */
  '\170', '\171', '\172', '\133', '\134', '\135', '\136', '\137',
  /* '`'     'a'     'b'     'c'     'd'     'e'     'f'     'g' */
  '\140', '\141', '\142', '\143', '\144', '\145', '\146', '\147',
  /* 'h'     'i'     'j'     'k'     'l'     'm'     'n'     'o' */
  '\150', '\151', '\152', '\153', '\154', '\155', '\156', '\157',
  /* 'p'     'q'     'r'     's'     't'     'u'     'v'     'w' */
  '\160', '\161', '\162', '\163', '\164', '\165', '\166', '\167',
  /* 'x'     'y'     'z'     '{'     '|'     '}'     '~' */
  '\170', '\171', '\172', '\173', '\174', '\175', '\176', '\177',

  /* Latin 1: */
  '\200', '\201', '\202', '\203', '\204', '\205', '\206', '\207',
  '\210', '\211', '\212', '\213', '\214', '\215', '\216', '\217',
  '\220', '\221', '\222', '\223', '\224', '\225', '\226', '\227',
  '\230', '\231', '\232', '\233', '\234', '\235', '\236', '\237',
  '\240', '\241', '\242', '\243', '\244', '\245', '\246', '\247',
  '\250', '\251', '\252', '\253', '\254', '\255', '\256', '\257',
  '\260', '\261', '\262', '\263', '\264', '\265', '\266', '\267',
  '\270', '\271', '\272', '\273', '\274', '\275', '\276', '\277',
  '\340', '\341', '\342', '\343', '\344', '\345', '\346', '\347',
  '\350', '\351', '\352', '\353', '\354', '\355', '\356', '\357',
  '\360', '\361', '\362', '\363', '\364', '\365', '\366', '\327',
  '\370', '\371', '\372', '\373', '\374', '\375', '\376', '\337',
  '\340', '\341', '\342', '\343', '\344', '\345', '\346', '\347',
  '\350', '\351', '\352', '\353', '\354', '\355', '\356', '\357',
  '\360', '\361', '\362', '\363', '\364', '\365', '\366', '\367',
  '\370', '\371', '\372', '\373', '\374', '\375', '\376', '\377',
};


static int
run_test (const char *pattern, struct re_registers *regs)
{
  static char text[] = "1111AAAA2222bbbb";

  struct re_pattern_buffer pat;

  const char *err;
  int res;
  int start2;

  memset (&pat, '\0', sizeof (pat));
  memset (regs, '\0', 2 * sizeof (regs[0]));
  pat.allocated = 0;		/* regex will allocate the buffer */
  pat.fastmap = (char *) malloc (256);
  if (pat.fastmap == NULL)
    {
      puts ("out of memory");
      exit (1);
    }

  pat.translate = (unsigned char *) casetable;

  err = re_compile_pattern (pattern, strlen (pattern), &pat);
  if (err != NULL)
    {
      fprintf (stderr, "/%s/: %s\n", pattern, err);
      exit (1);
    }
  res = re_search (&pat, text, strlen (text), 0, strlen (text), &regs[0]);
  if (res < 0)
    printf ("search 1: res = %d\n", res);
  else
    printf ("search 1: res = %d, start = %d, end = %d\n",
	    res, regs[0].start[0], regs[0].end[0]);

  if (regs[0].end == NULL)
    start2 = 8;
  else
    start2 = regs[0].end[0] + 1;
  regs[1] = regs[0];
  res = re_search (&pat, text, strlen (text), start2, strlen (text), &regs[1]);
  if (res < 0)
    printf ("search 2: res = %d\n", res);
  else
    printf ("search 2: res = %d, start = %d, end = %d\n",
	    res, regs[1].start[0], regs[1].end[0]);

  return res < 0 ? 1 : 0;
}


static int
do_test (void)
{
  static const char lower[] = "[[:lower:]]+";
  static const char upper[] = "[[:upper:]]+";
  struct re_registers regs[4];

  setlocale (LC_ALL, "C");

  (void) re_set_syntax (RE_SYNTAX_GNU_AWK);

  int result;
#define CHECK(exp) \
  if (exp) { puts (#exp); result = 1; }

  result = run_test (lower, regs);
  result |= run_test (upper, &regs[2]);
  if (! result)
    {
      CHECK (regs[0].start[0] != regs[2].start[0]);
      CHECK (regs[0].end[0] != regs[2].end[0]);
      CHECK (regs[1].start[0] != regs[3].start[0]);
      CHECK (regs[1].end[0] != regs[3].end[0]);
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
