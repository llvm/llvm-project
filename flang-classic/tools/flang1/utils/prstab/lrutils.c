/**
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file
 * \brief  LR utility functions
 *           - char conversion
 *           - file operations
 *           - miscellaneous
 */

#include "universal.h"
#include "lrutils.h"

void
a1tos1(INT *from, INT *to, INT count)
{
  INT i;
  INT n;

#ifdef TRACE
  (void)trace("a1tos1\n");
#endif

  if ((n = count) < 0)
    util_error("a1tos1 - bad count", FPS_ERR, ABORT);

  i = 0;
  while (n--) {
    to[i] = *(char *)(&from[i]); /* let c fetch find the char */
    i++;
  }

#ifdef TRACE
  (void)trace("a1tos1\n");
#endif
}

void
a4tos1(const char *from, INT *to, INT count)
{
  INT n;
  const char *a4; /* pointer used in letting C convert to S1 */
  INT *s1;

#ifdef TRACE
  (void)trace("a4tos1\n");
#endif

  if ((n = count) < 0)
    util_error("a4tos1 - bad count", FPS_ERR, ABORT);

  a4 = from;
  s1 = to;
  while (n--) {
    *s1++ = *a4++; /* let C do the byte swapping and unpacking */
  }

#ifdef TRACE
  (void)trace("a4tos1\n");
#endif
}

/**
 *   \brief Convert an AP 32-bit integer to unpacked right-justified
 *        ASCII character data (S1).
 *
 *   CALLING SEQUENCE:
 *       i32tos (from, to, count, sign, radix, format);
 *
 *   INPUT PARAMETERS:
 *       from            pointer to INT
 *                       (input parameter)
 *                       Specifies 32-bit integer quantity to be converted.
 *
 *       to              INT    array
 *                       (output parameter)
 *                       Specifies array containing converted S1
 *                       character data.
 *
 *       count           pointer to INT
 *                       (input parameter)
 *                       Specifies non-negative output field width for
 *                       conversion.
 *
 *       sign            pointer to INT
 *                       (input parameter)
 *                       Specifies whether signed conversion is used:
 *
 *                       0 = signed conversion.
 *                       1 = unsigned conversion.
 *
 *       radix           pointer to INT
 *                       (input parameter)
 *                       Specifies conversion radix:
 *
 *                        2 = binary radix.
 *                        8 = octal radix.
 *                       10 = decimal radix.
 *                       16 = hexadecimal radix.
 *
 *       format          pointer to INT
 *                       (input parameter)
 *                       Specifies conversion format:
 *
 *                       0 = left justified, blank filled.
 *                       1 = right justified, blank filled.
 *                       2 = right justified, zero filled.
 *
 *   SIDE EFFECTS:
 *       The function modifies the contents of the array specified
 *       by 'to'.
 *
 *   DESCRIPTION:
 *       Convert an AP 32-bit integer contained in the 'from' parameter
 *       to unpacked right-justified ASCII character data (S1) contained
 *       in the 'to' parameter.  The 'count' parameter specifies the number
 *       of characters of the to parameter that are filled with the
 *       converted result.  The 'sign' parameter indicates whether signed
 *       or unsigned conversion is to take place, negative numbers are
 *       preceded by a minus sign.  The 'radix' parameter refers to the
 *       radix to be used in conversion.  The 'format' parameter indicates
 *       where the conversion results will be placed in the output field
 *       and how unused space will be filled.  If the output field is
 *       too small for the result, the first character of the field is
 *       filled with an asterisk, and the low-order digits of the result
 *       fill the rest of the output field.
 *
 *   ERROR CONDITIONS:
 *       Illegal parameter value specified: user is notified and the
 *       program is aborted.
 *
 */

void
i32tos(INT *from, INT *to, INT count, INT sign, INT radix, INT format)
{
#define ASCII_OFFSET 48
#define ASTERISK 42
#define BLANK 32
#define HEX_OFFSET 7
#define LEGAL_FORMAT_SET 0x7
#define LEGAL_RADIX_SET 0x10504
#define LEGAL_SIGN_SET 0x3
#define MINUS 45
#define ZERO 48

  INT bit_width;     /* width of the bit field for a particular
                        radix */
  INT count_p;       /* pointer to the count of characters being
                           sent to a4tos1 (a4tos1 requires a pointer */
  INT idx;           /* for loop control variable */
  INT mask;          /* mask for bit_width of a particular radix */
  INT max_shift;     /* maximum number of bits from will need to be
                        shifted */
  INT msd;           /* index of the most-significant digit in to */
  INT num_bits;      /* number of bits to be shifted */
  UINT u_from;       /* used in shifting temp_from without sign
                        extension */
  INT temp_from;     /* temporary from (= abs(from)) */
  char temp_to[128]; /* termporary to - used to contain the
                        result returned from sprintf */

#ifdef TRACE
  (void)trace("i32tos\n");
#endif

  if (!INSET(LEGAL_SIGN_SET, sign))
    util_error("i32tos - bad sign", FPS_ERR, ABORT);

  if (!INSET(LEGAL_RADIX_SET, radix))
    util_error("i32tos - bad radix", FPS_ERR, ABORT);

  if (!INSET(LEGAL_FORMAT_SET, format))
    util_error("i32tos - bad format", FPS_ERR, ABORT);

  if (*from == 0) {
    msd = count - 1;
    to[msd] = ASCII_OFFSET;
  } else {
    if (sign == 0)
      temp_from = abs(*from);
    else
      temp_from = *from;

    if (radix == 10) {
      sprintf(temp_to, (sign == 0 ? "%d" : "%u"), temp_from);
      msd = count - strlen(temp_to);

      if (msd < 0) {
        a4tos1(&temp_to[abs(msd)], to, count);
        msd = -1;
      } else {
        count_p = count - msd;
        a4tos1(temp_to, &to[msd], count_p);
      }
    } else {
      switch (radix) {
      case 2:
        max_shift = 31;
        bit_width = 1;
        mask = 1;
        break;
      case 8:
        max_shift = 30;
        bit_width = 3;
        mask = 7;
        break;
      case 16:
        max_shift = 28;
        bit_width = 4;
        mask = 15;
        break;
      } /* end_switch */

      idx = count - 1;
      for (num_bits = 0; num_bits <= max_shift;
           num_bits = num_bits + bit_width) {
        if ((radix == 8) && (num_bits == 30))
          mask = 3;

        to[idx] = ASCII_OFFSET + ((temp_from >> num_bits) & mask);

        if (to[idx] != ASCII_OFFSET)
          msd = idx;

        if (to[idx] > '9')
          to[idx] = to[idx] + HEX_OFFSET;

        if (idx == 0) {
          u_from = temp_from;
          if ((u_from >> ((num_bits + bit_width) > 32 ? 32 : (num_bits +
                                                              bit_width))) != 0)
            msd = -1;
          break; /* out of for loop */
        } else
          idx = idx - 1;
        /* end_if_else */
      } /* end_for */
    }   /* end_if_else */
  }     /* end_if_else */

  if ((msd == -1) || ((sign == 0) && (*from < 0) && (msd == 0)))
    to[0] = ASTERISK;
  else {
    if (msd == 0)
      return;

    if (format == 0) {
      if ((sign == 0) && (*from < 0)) {
        msd = msd - 1;
        to[msd] = MINUS;
      }

      for (idx = msd; idx < count; ++idx)
        to[idx - msd] = to[idx];

      for (idx = count - msd; idx < count; ++idx)
        to[idx] = BLANK;
    } else {
      for (idx = 0; idx < msd; ++idx)
        to[idx] = (format == 1 ? BLANK : ZERO);

      if ((sign == 0) && (*from < 0)) {
        if (format == 1) {
          to[msd - 1] = MINUS;
        } else {
          to[0] = MINUS;
        }
      }
    } /* end_if_else */
  }   /* end_if_else */

#ifdef TRACE
  (void)trace("i32tos\n");
#endif
}

void
s1toa1(INT *from, INT *to, INT *count)
{
  INT i; /* for loop control variable */

  if (*count < 0)
    util_error("s1toa1 - bad count", FPS_ERR, ABORT);

  for (i = 0; i < (*count); ++i) {
    to[i] = 0x20202020;
    *((char *)&to[i]) = from[i];
  }
}

void
s1toa4(INT *from, INT *to, INT *count)
{
  INT n;
  char *a4; /* pointer used in letting C convert to A4 */
  INT *s1;

#ifdef TRACE
  (void)trace("s1toa4\n");
#endif

  if ((n = *count) < 0)
    util_error("s1toa4 - bad count", FPS_ERR, ABORT);

  a4 = (char *)to;
  s1 = from;

  while (n--)
    *a4++ = *s1++;

  /**
   * pad with blanks to the end of the word  -  switch value is the number
   * of bytes written in the last word
   */
  switch (*count & 3) {
  case 0:
    break;
  case 1:
    *a4++ = ' ';
    FLANG_FALLTHROUGH;
  case 2:
    *a4++ = ' ';
    FLANG_FALLTHROUGH;
  case 3:
    *a4++ = ' ';
  }

#ifdef TRACE
  (void)trace("s1toa4\n");
#endif
}

/**
 *
 *   \brief Find specified argument in switch table.
 *
 *   CALLING SEQUENCE:
 *       ret_arg = find (swtab, arg_table, cur_arg);
 *
 *   INPUT PARAMETERS:
 *       swtab           INT    array
 *                       (input parameter)
 *                       Switch Table.
 *
 *       arg_table       INT    array
 *                       (input parameter)
 *                       Argument table.
 *
 *       cur_arg         INT    scalar
 *                       (input parameter)
 *                       Specifies argument to search for in switch table.
 *
 *   RETURN VALUE:
 *       type            INT
 *                       Number specifying the type of the argrment.
 *
 *                       > 0    argument type from 'swtab'
 *                       = 0    argument not found in 'swtab'
 *
 *   DESCRIPTION:
 *       'swtab' is searched for the argument in 'arg_table' pointed
 *       to by 'cur_arg'. If the argument is found, the switch type of
 *       the argument is returned. If the argument isn't found 0 is
 *       returned.
 *
 */

INT
find(INT *swtab, INT arg_table[ARG_TABLE_ROWS][ARG_TABLE_COLS], INT cur_arg)
{

  INT char_cnt;       /* count of the number of characters compared */
  INT cur_switch_ptr; /* index of the switch currently being compared
                         against */
  INT found;          /* flag for if the element in arg_table indicated
                         by cur_arg has been found in swtab */
  INT match;          /* flag for if the element in arg_table indicated
                         by cur_arg can match the current element in
                         swtab being looked at */
  INT offset;         /* offset used in processing a '.' in an swtab entry */

  found = FALSE;
  cur_switch_ptr = 0;

  while ((!found) && (swtab[cur_switch_ptr] != END_OF_TABLE_MARK)) {
    if (swtab[cur_switch_ptr + 1] >= arg_table[cur_arg][0]) {
      match = TRUE;
      offset = 1;
      char_cnt = 1;

      while (match && (char_cnt <= arg_table[cur_arg][0])) {
        if (swtab[cur_switch_ptr + offset + char_cnt] == '.') {
          offset = 2;
        } /* end_if */

        if (swtab[cur_switch_ptr + offset + char_cnt] ==
            arg_table[cur_arg][char_cnt]) {
          char_cnt = char_cnt + 1;
        }

        else {
          match = FALSE;
        } /* end_if_else */

      } /* end_while */

      if (match) {
        if (offset == 1) {
          if ((swtab[cur_switch_ptr + char_cnt + 1] == '.') ||
              (swtab[cur_switch_ptr + 1] == arg_table[cur_arg][0])) {
            found = TRUE;
          } else {
            found = FALSE;
          } /* end_if_else */
        } else {
          if (swtab[cur_switch_ptr + 1] >= char_cnt) {
            found = TRUE;
          } /* end_if */

        } /* end_if_else */

      } /* end_if */

    } /* end_if */

    if (!found) {
      cur_switch_ptr = cur_switch_ptr + 2 + swtab[cur_switch_ptr + 1];
    } /* end_if */

  } /* end_while */

  if (found) {
    return (swtab[cur_switch_ptr]);
  } else {
    return (0);
  } /* end_if_else */
}

/**
 *
 *   \brief Read a line from a file.
 *
 *   CALLING SEQUENCE:
 *       status = rdlinf (fp, buffer, count)
 *
 *   INPUT PARAMETERS:
 *        fp              pointer to long
 *                        (input parameter)
 *                        Specifies file descriptor.
 *
 *        count           pointer to long
 *                        (input parameter)
 *                        Specifies maximum count of characters to read
 *                        (must be non-negative)
 *
 *        buffer          long       array
 *                        (output parameter)
 *                        Specifies characters read in S1 format.
 *
 *   RETURN VALUE:
 *       status           long
 *                        Specifies completion status of read operation:
 *
 *                        -1  = end-of-file, no line read
 *                        >=0 = count of characters read
 *
 *   SIDE EFFECTS:
 *       This function modifies the contents of the array specified
 *       by 'buffer'.
 *
 *   DESCRIPTION:
 *       Read a line from file indicated by 'fp'.
 *       Characters are read up to end-of-line or
 *       end-of-file, whichever comes first. If the
 *       line is longer that the maximum specified by
 *       'count', the extra characters are discarded
 *       without error. All trailing blanks on a line
 *       are removed, so the value of status is the
 *       number of characters read, discarding trailing
 *       blanks (and trailing tabs if applicable).
 *
 *   ERROR CONDITIONS:
 *       Illegal parameter value specified or error in making
 *       a sytem call: user is notified and the program is
 *       aborted.
 *
 */

INT
rdline(FILE *fp, INT *buffer, INT count)
{
  char *fgets_status;             /* status returned by call to fgets */
  INT num_char;                   /* number of characters in temp_buffer */
  INT status;                     /* status returned by call to fseek */
  char temp_buffer[MAX_LINE_LEN]; /* temporary buffer, hold buffer in
                                     A4 format */

#ifdef TRACE
  (void)trace("rdline\n");
#endif

  if (!fp)
    util_error("rdline - bad fp", FPS_ERR, ABORT);

  if (count < 0)
    util_error("rdline - bad count", FPS_ERR, ABORT);

  fgets_status = fgets(temp_buffer, MAX_LINE_LEN + 1, fp);
  if (fgets_status == NULL) {
    status = feof(fp);
    if (status == 0)
      util_error("rdline - fgets failed", SYS_ERR, ABORT);
    else
      return (-1);
  }

  num_char = strlen(temp_buffer) - 1;
  if (num_char > count)
    num_char = count;
  a4tos1(temp_buffer, buffer, num_char);

  return (num_char);

#ifdef TRACE
  (void)trace("rdline\n");
#endif
}

void
wtline(FILE *fp, INT *buffer, INT count)
{
  INT status;                     /* status returned by a system call */
  char temp_buffer[MAX_LINE_LEN]; /* temporary buffer */
  INT n;

#ifdef TRACE
  (void)trace("wtline\n");
#endif

  if (!fp) {
    util_error("wtline - bad fp", FPS_ERR, ABORT);
  }
  if ((n = count) < 0) {
    util_error("wtline - bad count", FPS_ERR, ABORT);
  }

  s1toa4(buffer, (INT *)temp_buffer, &n);

  while (n--)
    if (temp_buffer[n] != ' ') /* strip trailing blanks */
      break;

  temp_buffer[n + 1] = '\n';
  temp_buffer[n + 2] = '\0';

  status = fprintf(fp, "%s", temp_buffer);
  if (status < 0)
    util_error("wtline - fprint failed", SYS_ERR, ABORT);

#ifdef TRACE
  (void)trace("wtline\n");
#endif
}

/**
 *   \brief Write a page mark to a file.
 *
 *   CALLING SEQUENCE:
 *       wtpage (fd)
 *
 *   INPUT PARAMETERS:
 *       fd              pointer to long
 *                       (input parameter)
 *                       Specifies file descriptor.
 *
 *   RETURN VALUE:
 *       NONE.
 *
 *   SIDE EFFECTS:
 *       NONE.
 *
 *   DESCRIPTION:
 *       Write a page marker to file indicated by 'fd'. A page marker
 *       is a host-dependent indication in a text file which causes a
 *       skip to the top of the next page when the file is printed.
 *
 *   ERROR CONDITIONS:
 *       Illegal parameter value, illegal file type, illegal open
 *       access, or error in making a system call: the user is notified
 *       and the program is aborted.
 *
 */

INT
wtpage(FILE *fp)
{
#define FORM_FEED "\f\n"

  INT status; /* status returned by a system call */

#ifdef TRACE
  (void)trace("wtpage\n");
#endif

  if (!fp)
    util_error("wtpage - bad fp", FPS_ERR, ABORT);

  status = fseek(fp, 0, 1);
  if (status != FSEEK_OK)
    util_error("wtpage - fseek failed", SYS_ERR, ABORT);

  status = fputs(FORM_FEED, fp);
  if (status == EOF)
    util_error("wtpage - fputs failed", SYS_ERR, ABORT);

#ifdef TRACE
  (void)trace("wtpage\n");
#endif

  return 0;
}

/**
 *   \brief Add command line argument(s) (switch) into 'argtab'.
 *
 *   CALLING SEQUENCE:
 *       add_arg_entry (argtab, arglen, argtab_end_ptr, arg_table, type,
 *                      first_lbl, lbl_cnt);
 *
 *   INPUT PARAMETERS:
 *       argtab          long    array
 *                        (output parameter)
 *                        Table in which command line arguments are
 *                        returned.
 *
 *       arglen          long    scalar
 *                        (input parameter)
 *                        Length of 'argtab'.
 *
 *       argtab_end_ptr  pointer to long
 *                        (input/output parameter)
 *                        Pointer to the last element in 'argtab'.
 *
 *       arg_table       long    array
 *                        (input parameter)
 *                        Table of argruments on the command line.
 *
 *       type            long    scalar
 *                        (input parameter)
 *                        Type of the switch being added to 'argtab'.
 *
 *       first_lbl       long    scalar
 *                        (input parameter)
 *                        Pointer the first element in 'arg_table' to
 *                        be added to 'argtab'.
 *
 *       lbl_cnt         long    scalar
 *                        (input parameter)
 *                        Count of the number of switches to be added to
 *                        'argtab'.
 *
 *   RETURN VALUE:
 *       NONE.
 *
 *   SIDE EFFECTS:
 *       This function modifies the contents of the array specified
 *       by 'argtab'. This function also modifies the contents of the
 *       long specified by 'argtab_len_ptr'.
 *
 *   DESCRIPTION:
 *       'lbl_cnt' command line arguments stored in 'arg_table', starting
 *       with the switch pointed to by 'first_lbl', are added to 'arg_tab'.
 *       'argtab_end_ptr' is set to point to the last element in 'argtab'.
 *       If table overflow occurs 'argtab_end_ptr' is set to -3.
 *
 */

void
add_arg_entry(INT *argtab, INT arglen, INT *argtab_end_ptr,
              INT arg_table[ARG_TABLE_ROWS][ARG_TABLE_COLS], INT type,
              INT first_lbl, INT lbl_cnt)
{

  INT cur_char;       /* index of character being processed */
  INT cur_lbl;        /* index of labels being processed */
  INT total_elements; /* total number of elements to be added to
                         argtab */

  total_elements = 0;
  for (cur_lbl = first_lbl; cur_lbl <= (first_lbl + lbl_cnt - 1); cur_lbl++) {
    total_elements = total_elements + arg_table[cur_lbl][0] + 1;
  }

  if ((total_elements + *argtab_end_ptr) >= (arglen - 1)) {
    argtab[*argtab_end_ptr] = OVERFLOW_MARK;
    *argtab_end_ptr = OVERFLOW_MARK;
  } else {
    *argtab_end_ptr = *argtab_end_ptr + 1;
    argtab[*argtab_end_ptr] = type;
    *argtab_end_ptr = *argtab_end_ptr + 1;
    argtab[*argtab_end_ptr] = lbl_cnt;

    for (cur_lbl = first_lbl; cur_lbl <= (first_lbl + lbl_cnt - 1); cur_lbl++) {
      *argtab_end_ptr = *argtab_end_ptr + 1;
      argtab[*argtab_end_ptr] = arg_table[cur_lbl][0];
      for (cur_char = 1; cur_char <= (arg_table[cur_lbl][0]); cur_char++) {
        *argtab_end_ptr = *argtab_end_ptr + 1;
        argtab[*argtab_end_ptr] = arg_table[cur_lbl][cur_char];
      } /* end_for */
    }   /* end_for */
  }     /* end_if_else */
}

static char argbuf[100];

void
xargvar(INT num)
{
  char *p;
  INT k;

  p = argbuf;
  k = num;
  /**
   * getarg fills n positions in argbuf (n = sizeof argbuf - 1).
   * the trailing blanks are deleted (the first blank after the
   * argument is replaced with a null char
   */
  getarg(&k, argbuf, (INT)(sizeof(argbuf) - 1));
  p = &argbuf[sizeof(argbuf) - 1];
  while (*--p == ' ')
    ;
  *(p + 1) = '\0';
}

void
cli(INT *swtab, INT *argtab, INT arglen, INT caller)
{
  INT arg_table[ARG_TABLE_ROWS][ARG_TABLE_COLS];
  /**
   * Table of arguments on the command line. Each row contains a single
   * argument. The first elemement of each row is equal to the number of
   * characters in the argument. The remaining row elements contain the
   * characters of the argument in S1 format
   */
  INT argtab_end_ptr; /* Pointer to the last element in
                       * argtab */
  INT cur_arg;        /* Row number to the argument in arg_table
                       * currently being processed */
  INT dash = 0x2d;    /* Used to identify a switch */
  INT i;              /* Loop control variables */
  INT num_args;       /* Number of arguments in arg_table */
  INT num_lbls;       /* Number of labels grouped with a switch */
  INT type;           /* Type for the switch being processed */

  if (arglen <= 1) {
    argtab[0] = OVERFLOW_MARK; /* table overflow */
    return;
  } /* end_if */
  argtab_end_ptr = -1;
  argtab[argtab_end_ptr] = END_OF_TABLE_MARK;

  /* retrieve command line argruments */

  num_args = iargc();
  for (cur_arg = 0; cur_arg < num_args; cur_arg++) {
    xargvar(cur_arg + 1);
    arg_table[cur_arg][0] = strlen(argbuf);
    a4tos1(argbuf, &arg_table[cur_arg][1], arg_table[cur_arg][0]);
    lower_to_upper(cur_arg, arg_table);
  } /* end_for */

  /**
   * processing loop which fills argtab with the command line arguments
   */

  cur_arg = 0;
  while (cur_arg < num_args) {
    /* determine if agrument is switched */
    if (arg_table[cur_arg][1] == dash) {

      /* find out how many labels are grouped with this switch */

      num_lbls = 1;
      while ((cur_arg + num_lbls <= num_args - 1) &&
             (arg_table[cur_arg + num_lbls][1] != dash))
        num_lbls++;
      /* end_while */

      /* strip the '-' from the switch */

      for (i = 0; i < arg_table[cur_arg][0]; ++i)
        arg_table[cur_arg][i + 1] = arg_table[cur_arg][i + 2];
      --arg_table[cur_arg][0];

      /* set the type of the switch being processed */

      type = find(swtab, arg_table, cur_arg);

      /*
       * stuff the switch and the labels grouped with it into argtab
       */

      if (type > 0)
        add_arg_entry(argtab, arglen, &argtab_end_ptr, arg_table, type,
                      cur_arg + 1, num_lbls - 1);
      else
        add_arg_entry(argtab, arglen, &argtab_end_ptr, arg_table,
                      UNEXPECTED_ARG_MARK, cur_arg, num_lbls);
      /* end_if_else */

      if (argtab_end_ptr == OVERFLOW_MARK)
        return; /* table overflow */

      /* increment cur_arg to the next argument */

      cur_arg = cur_arg + num_lbls;
    } else {
      /* stuff the unswithced argements into argtab */

      add_arg_entry(argtab, arglen, &argtab_end_ptr, arg_table,
                    UNSWITCHED_ARG_MARK, cur_arg, 1);

      if (argtab_end_ptr == OVERFLOW_MARK)
        return; /* table overflow */

      cur_arg++;

    } /* end_if_else */
  }   /* end_while */
}

void
exitf(INT errcnd)
{

  INT cond_originator; /* condition originator code */
  INT error_level;     /* error level severity code */

  error_level = 0;
  cond_originator = 0;
  error_level = ((unsigned)errcnd >> 29) & 0x0007;
  cond_originator = ((unsigned)errcnd >> 12) & 0x0FFF;

  printf("completion code: ");
  switch (error_level) {
  case 0:
    printf("success ");
    break;
  case 2:
    printf("warning ");
    break;
  case 5:
    printf("error ");
    break;
  case 6:
    printf("severe ");
    break;
  case 7:
    printf("terminal ");
    break;
  default:
    printf("invalid completion code ");
    break;
  } /* end_switch */

  switch (cond_originator) {
  case 100:
    printf("(apal64)\n");
    break;
  case 101:
    printf("(aplink64\n");
    break;
  case 102:
    printf("(apftn64)\n");
    break;
  case 103:
    printf("(aplibr64)\n");
    break;
  case 104:
    printf("(apdebug64)\n");
    break;
  case 105:
    printf("(util64)\n");
    break;
  case 200:
    printf("(apex)\n");
    break;
  case 201:
    printf("(sum)\n");
    break;
  case 202:
    printf("(aprq)\n");
    break;
  case 203:
    printf("(trap)\n");
    break;
  case 204:
    printf("(jsys)\n");
    break;
  case 205:
    printf("(dsmr)\n");
    break;
  case 206:
    printf("(ldsup)\n");
    break;
  case 300:
    printf("(sys)\n");
    break;
  case 301:
    printf("(for)\n");
    break;
  case 302:
    printf("(utl)\n");
    break;
  case 303:
    printf("(fif)\n");
    break;
  case 304:
    printf("(aop)\n");
    break;
  case 305:
    printf("(mth)\n");
    break;
  case 306:
    printf("(d64)\n");
    break;
  case 307:
    printf("(sje i/o)\n");
    break;
  default:
    printf("\n");
    break;
  } /* end_switch */

  exit(error_level);
}

/**
 * subroutine getarg(k, c)
 * returns the kth unix command argument in fortran character
 * variable argument c
 */

void
getarg(INT *n, char *s, INT ls)
{
  extern INT xargc;
  extern char **xargv;
  const char *t;
  INT i;

  if (*n >= 0 && *n < xargc)
    t = xargv[*n];
  else
    t = "";

  for (i = 0; i < ls && *t != '\0'; ++i)
    *s++ = *t++;
  for (; i < ls; ++i)
    *s++ = ' ';
}

INT
iargc(void)
{
  extern INT xargc;
  return (xargc - 1);
}

/**
 *
 *   \brief Convert the specified argument in the argument table
 *       from lower case to upper case.
 *
 *   CALLING SEQUENCE:
 *       lower_to_upper(cur_arg, arg_table);
 *
 *   INPUT PARAMETERS:
 *       cur_arg         long    scalar
 *                       (input parameter)
 *                       Index into the argument table, indicating the
 *                       argument to be converted from lower case to
 *                       upper case.
 *
 *       arg_table       long    array
 *                       (output parameter)
 *                       Argument table, containing the argument to be
 *                       converted from lower case to upper case.
 *
 *   RETURN VALUE:
 *       NONE.
 *
 *   SIDE EFFECTS:
 *       NONE.
 *
 *   DESCRIPTION:
 *       Convert the specified entry in the argument table from
 *       lower case to upper case.
 *
 *   ERROR CONDITIONS:
 *       NONE.
 *
 */

void
lower_to_upper(INT cur_arg, INT arg_table[ARG_TABLE_ROWS][ARG_TABLE_COLS])
{
#define OFFSET 0x20

  INT col; /* index into column of character to be converted */

  for (col = 1; col <= (arg_table[cur_arg][0]); col++) {
    /**
     * If the character is in lower case, then convert it to
     * upper case.
     */

    if ((arg_table[cur_arg][col] >= 'a') && (arg_table[cur_arg][col] <= 'z')) {
      arg_table[cur_arg][col] -= OFFSET;
    } /* end_if */
  }   /* end_for */
}

void
util_error(const char *p, INT err, INT flag)
{
  printf("%s\n", p);
  exit(0);
}

void
trace(char *p)
{
  printf("%s\n", p);
}

/**
 *   \brief This function converts the charactets of a file name from upper
 *   to lower case.
 *
 *   This function recieves two parameters: name - a file name in S1
 *   format; col - the column in name that is to be converted to lower
 *   case. The conversion is performed by adding the ascii offset to
 *   the letter in name[col] if it is an upper case letter.
 *
 */

INT
upper_to_lower(INT *name, INT col)
{
#define OFFSET 0x20

  if ((name[col] >= 'A') && (name[col] <= 'Z'))
    return (name[col] + OFFSET);
  else
    return (name[col]);
}
