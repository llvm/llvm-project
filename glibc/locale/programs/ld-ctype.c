/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.org>, 1995.

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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <alloca.h>
#include <byteswap.h>
#include <endian.h>
#include <errno.h>
#include <limits.h>
#include <obstack.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <wctype.h>
#include <stdint.h>
#include <sys/uio.h>

#include "localedef.h"
#include "charmap.h"
#include "localeinfo.h"
#include "langinfo.h"
#include "linereader.h"
#include "locfile-token.h"
#include "locfile.h"

#include <assert.h>


/* The bit used for representing a special class.  */
#define BITPOS(class) ((class) - tok_upper)
#define BIT(class) (_ISbit (BITPOS (class)))
#define BITw(class) (_ISwbit (BITPOS (class)))

#define ELEM(ctype, collection, idx, value)				      \
  *find_idx (ctype, &ctype->collection idx, &ctype->collection##_max idx,     \
	     &ctype->collection##_act idx, value)


/* To be compatible with former implementations we for now restrict
   the number of bits for character classes to 16.  When compatibility
   is not necessary anymore increase the number to 32.  */
#define char_class_t uint16_t
#define char_class32_t uint32_t


/* Type to describe a transliteration action.  We have a possibly
   multiple character from-string and a set of multiple character
   to-strings.  All are 32bit values since this is what is used in
   the gconv functions.  */
struct translit_to_t
{
  uint32_t *str;

  struct translit_to_t *next;
};

struct translit_t
{
  uint32_t *from;

  const char *fname;
  size_t lineno;

  struct translit_to_t *to;

  struct translit_t *next;
};

struct translit_ignore_t
{
  uint32_t from;
  uint32_t to;
  uint32_t step;

  const char *fname;
  size_t lineno;

  struct translit_ignore_t *next;
};


/* Type to describe a transliteration include statement.  */
struct translit_include_t
{
  const char *copy_locale;
  const char *copy_repertoire;

  struct translit_include_t *next;
};

/* Provide some dummy pointer for empty string.  */
static uint32_t no_str[] = { 0 };


/* Sparse table of uint32_t.  */
#define TABLE idx_table
#define ELEMENT uint32_t
#define DEFAULT ((uint32_t) ~0)
#define NO_ADD_LOCALE
#include "3level.h"

#define TABLE wcwidth_table
#define ELEMENT uint8_t
#define DEFAULT 0xff
#include "3level.h"

#define TABLE wctrans_table
#define ELEMENT int32_t
#define DEFAULT 0
#define wctrans_table_add wctrans_table_add_internal
#include "3level.h"
#undef wctrans_table_add
/* The wctrans_table must actually store the difference between the
   desired result and the argument.  */
static inline void
wctrans_table_add (struct wctrans_table *t, uint32_t wc, uint32_t mapped_wc)
{
  wctrans_table_add_internal (t, wc, mapped_wc - wc);
}

/* Construction of sparse 3-level tables.
   See wchar-lookup.h for their structure and the meaning of p and q.  */

struct wctype_table
{
  /* Parameters.  */
  unsigned int p;
  unsigned int q;
  /* Working representation.  */
  size_t level1_alloc;
  size_t level1_size;
  uint32_t *level1;
  size_t level2_alloc;
  size_t level2_size;
  uint32_t *level2;
  size_t level3_alloc;
  size_t level3_size;
  uint32_t *level3;
  size_t result_size;
};

static void add_locale_wctype_table (struct locale_file *file,
				     struct wctype_table *t);

/* The real definition of the struct for the LC_CTYPE locale.  */
struct locale_ctype_t
{
  uint32_t *charnames;
  size_t charnames_max;
  size_t charnames_act;
  /* An index lookup table, to speedup find_idx.  */
  struct idx_table charnames_idx;

  struct repertoire_t *repertoire;

  /* We will allow up to 8 * sizeof (uint32_t) character classes.  */
#define MAX_NR_CHARCLASS (8 * sizeof (uint32_t))
  size_t nr_charclass;
  const char *classnames[MAX_NR_CHARCLASS];
  uint32_t last_class_char;
  uint32_t class256_collection[256];
  uint32_t *class_collection;
  size_t class_collection_max;
  size_t class_collection_act;
  uint32_t class_done;
  uint32_t class_offset;

  struct charseq **mbdigits;
  size_t mbdigits_act;
  size_t mbdigits_max;
  uint32_t *wcdigits;
  size_t wcdigits_act;
  size_t wcdigits_max;

  struct charseq *mboutdigits[10];
  uint32_t wcoutdigits[10];
  size_t outdigits_act;

  /* If the following number ever turns out to be too small simply
     increase it.  But I doubt it will.  --drepper@gnu */
#define MAX_NR_CHARMAP 16
  const char *mapnames[MAX_NR_CHARMAP];
  uint32_t *map_collection[MAX_NR_CHARMAP];
  uint32_t map256_collection[2][256];
  size_t map_collection_max[MAX_NR_CHARMAP];
  size_t map_collection_act[MAX_NR_CHARMAP];
  size_t map_collection_nr;
  size_t last_map_idx;
  int tomap_done[MAX_NR_CHARMAP];
  uint32_t map_offset;

  /* Transliteration information.  */
  struct translit_include_t *translit_include;
  struct translit_t *translit;
  struct translit_ignore_t *translit_ignore;
  uint32_t ntranslit_ignore;

  uint32_t *default_missing;
  const char *default_missing_file;
  size_t default_missing_lineno;

  uint32_t to_nonascii;
  uint32_t nonascii_case;

  /* The arrays for the binary representation.  */
  char_class_t *ctype_b;
  char_class32_t *ctype32_b;
  uint32_t **map_b;
  uint32_t **map32_b;
  uint32_t **class_b;
  struct wctype_table *class_3level;
  struct wctrans_table *map_3level;
  uint32_t *class_name_ptr;
  uint32_t *map_name_ptr;
  struct wcwidth_table width;
  uint32_t mb_cur_max;
  const char *codeset_name;
  uint32_t *translit_from_idx;
  uint32_t *translit_from_tbl;
  uint32_t *translit_to_idx;
  uint32_t *translit_to_tbl;
  uint32_t translit_idx_size;
  size_t translit_from_tbl_size;
  size_t translit_to_tbl_size;

  struct obstack mempool;
};


/* Marker for an empty slot.  This has the value 0xFFFFFFFF, regardless
   whether 'int' is 16 bit, 32 bit, or 64 bit.  */
#define EMPTY ((uint32_t) ~0)


#define obstack_chunk_alloc xmalloc
#define obstack_chunk_free free


/* Prototypes for local functions.  */
static void ctype_startup (struct linereader *lr, struct localedef_t *locale,
			   const struct charmap_t *charmap,
			   struct localedef_t *copy_locale,
			   int ignore_content);
static void ctype_class_new (struct linereader *lr,
			     struct locale_ctype_t *ctype, const char *name);
static void ctype_map_new (struct linereader *lr,
			   struct locale_ctype_t *ctype,
			   const char *name, const struct charmap_t *charmap);
static uint32_t *find_idx (struct locale_ctype_t *ctype, uint32_t **table,
			   size_t *max, size_t *act, uint32_t idx);
static void set_class_defaults (struct locale_ctype_t *ctype,
				const struct charmap_t *charmap,
				struct repertoire_t *repertoire);
static void allocate_arrays (struct locale_ctype_t *ctype,
			     const struct charmap_t *charmap,
			     struct repertoire_t *repertoire);


static const char *longnames[] =
{
  "zero", "one", "two", "three", "four",
  "five", "six", "seven", "eight", "nine"
};
static const char *uninames[] =
{
  "U00000030", "U00000031", "U00000032", "U00000033", "U00000034",
  "U00000035", "U00000036", "U00000037", "U00000038", "U00000039"
};
static const unsigned char digits[] = "0123456789";


static void
ctype_startup (struct linereader *lr, struct localedef_t *locale,
	       const struct charmap_t *charmap,
	       struct localedef_t *copy_locale, int ignore_content)
{
  unsigned int cnt;
  struct locale_ctype_t *ctype;

  if (!ignore_content && locale->categories[LC_CTYPE].ctype == NULL)
    {
      if (copy_locale == NULL)
	{
	  /* Allocate the needed room.  */
	  locale->categories[LC_CTYPE].ctype = ctype =
	    (struct locale_ctype_t *) xcalloc (1,
					       sizeof (struct locale_ctype_t));

	  /* We have seen no names yet.  */
	  ctype->charnames_max = charmap->mb_cur_max == 1 ? 256 : 512;
	  ctype->charnames = (uint32_t *) xmalloc (ctype->charnames_max
						   * sizeof (uint32_t));
	  for (cnt = 0; cnt < 256; ++cnt)
	    ctype->charnames[cnt] = cnt;
	  ctype->charnames_act = 256;
	  idx_table_init (&ctype->charnames_idx);

	  /* Fill character class information.  */
	  ctype->last_class_char = ILLEGAL_CHAR_VALUE;
	  /* The order of the following instructions determines the bit
	     positions!  */
	  ctype_class_new (lr, ctype, "upper");
	  ctype_class_new (lr, ctype, "lower");
	  ctype_class_new (lr, ctype, "alpha");
	  ctype_class_new (lr, ctype, "digit");
	  ctype_class_new (lr, ctype, "xdigit");
	  ctype_class_new (lr, ctype, "space");
	  ctype_class_new (lr, ctype, "print");
	  ctype_class_new (lr, ctype, "graph");
	  ctype_class_new (lr, ctype, "blank");
	  ctype_class_new (lr, ctype, "cntrl");
	  ctype_class_new (lr, ctype, "punct");
	  ctype_class_new (lr, ctype, "alnum");

	  ctype->class_collection_max = charmap->mb_cur_max == 1 ? 256 : 512;
	  ctype->class_collection
	    = (uint32_t *) xcalloc (sizeof (unsigned long int),
				    ctype->class_collection_max);
	  ctype->class_collection_act = 256;

	  /* Fill character map information.  */
	  ctype->last_map_idx = MAX_NR_CHARMAP;
	  ctype_map_new (lr, ctype, "toupper", charmap);
	  ctype_map_new (lr, ctype, "tolower", charmap);

	  /* Fill first 256 entries in `toXXX' arrays.  */
	  for (cnt = 0; cnt < 256; ++cnt)
	    {
	      ctype->map_collection[0][cnt] = cnt;
	      ctype->map_collection[1][cnt] = cnt;

	      ctype->map256_collection[0][cnt] = cnt;
	      ctype->map256_collection[1][cnt] = cnt;
	    }

	  if (enc_not_ascii_compatible)
	    ctype->to_nonascii = 1;

	  obstack_init (&ctype->mempool);
	}
      else
	ctype = locale->categories[LC_CTYPE].ctype =
	  copy_locale->categories[LC_CTYPE].ctype;
    }
}


void
ctype_finish (struct localedef_t *locale, const struct charmap_t *charmap)
{
  /* See POSIX.2, table 2-6 for the meaning of the following table.  */
#define NCLASS 12
  static const struct
  {
    const char *name;
    const char allow[NCLASS];
  }
  valid_table[NCLASS] =
  {
    /* The order is important.  See token.h for more information.
       M = Always, D = Default, - = Permitted, X = Mutually exclusive  */
    { "upper",  "--MX-XDDXXX-" },
    { "lower",  "--MX-XDDXXX-" },
    { "alpha",  "---X-XDDXXX-" },
    { "digit",  "XXX--XDDXXX-" },
    { "xdigit", "-----XDDXXX-" },
    { "space",  "XXXXX------X" },
    { "print",  "---------X--" },
    { "graph",  "---------X--" },
    { "blank",  "XXXXXM-----X" },
    { "cntrl",  "XXXXX-XX--XX" },
    { "punct",  "XXXXX-DD-X-X" },
    { "alnum",  "-----XDDXXX-" }
  };
  size_t cnt;
  int cls1, cls2;
  uint32_t space_value;
  struct charseq *space_seq;
  struct locale_ctype_t *ctype = locale->categories[LC_CTYPE].ctype;
  int warned;
  const void *key;
  size_t len;
  void *vdata;
  void *curs;

  /* Now resolve copying and also handle completely missing definitions.  */
  if (ctype == NULL)
    {
      const char *repertoire_name;

      /* First see whether we were supposed to copy.  If yes, find the
	 actual definition.  */
      if (locale->copy_name[LC_CTYPE] != NULL)
	{
	  /* Find the copying locale.  This has to happen transitively since
	     the locale we are copying from might also copying another one.  */
	  struct localedef_t *from = locale;

	  do
	    from = find_locale (LC_CTYPE, from->copy_name[LC_CTYPE],
				from->repertoire_name, charmap);
	  while (from->categories[LC_CTYPE].ctype == NULL
		 && from->copy_name[LC_CTYPE] != NULL);

	  ctype = locale->categories[LC_CTYPE].ctype
	    = from->categories[LC_CTYPE].ctype;
	}

      /* If there is still no definition issue an warning and create an
	 empty one.  */
      if (ctype == NULL)
	{
	  record_warning (_("\
No definition for %s category found"), "LC_CTYPE");
	  ctype_startup (NULL, locale, charmap, NULL, 0);
	  ctype = locale->categories[LC_CTYPE].ctype;
	}

      /* Get the repertoire we have to use.  */
      repertoire_name = locale->repertoire_name ?: repertoire_global;
      if (repertoire_name != NULL)
	ctype->repertoire = repertoire_read (repertoire_name);
    }

  /* We need the name of the currently used 8-bit character set to
     make correct conversion between this 8-bit representation and the
     ISO 10646 character set used internally for wide characters.  */
  ctype->codeset_name = charmap->code_set_name;
  if (ctype->codeset_name == NULL)
    {
      record_error (0, 0, _("\
No character set name specified in charmap"));
      ctype->codeset_name = "//UNKNOWN//";
    }

  /* Set default value for classes not specified.  */
  set_class_defaults (ctype, charmap, ctype->repertoire);

  /* Check according to table.  */
  for (cnt = 0; cnt < ctype->class_collection_act; ++cnt)
    {
      uint32_t tmp = ctype->class_collection[cnt];

      if (tmp != 0)
	{
	  for (cls1 = 0; cls1 < NCLASS; ++cls1)
	    if ((tmp & _ISwbit (cls1)) != 0)
	      for (cls2 = 0; cls2 < NCLASS; ++cls2)
		if (valid_table[cls1].allow[cls2] != '-')
		  {
		    int eq = (tmp & _ISwbit (cls2)) != 0;
		    switch (valid_table[cls1].allow[cls2])
		      {
		      case 'M':
			if (!eq)
			  {
			    uint32_t value = ctype->charnames[cnt];

			    record_error (0, 0, _("\
character L'\\u%0*x' in class `%s' must be in class `%s'"),
					  value > 0xffff ? 8 : 4,
					  value,
					  valid_table[cls1].name,
					  valid_table[cls2].name);
			  }
			break;

		      case 'X':
			if (eq)
			  {
			    uint32_t value = ctype->charnames[cnt];

			    record_error (0, 0, _("\
character L'\\u%0*x' in class `%s' must not be in class `%s'"),
					  value > 0xffff ? 8 : 4,
					  value,
					  valid_table[cls1].name,
					  valid_table[cls2].name);
			  }
			break;

		      case 'D':
			ctype->class_collection[cnt] |= _ISwbit (cls2);
			break;

		      default:
			record_error (5, 0, _("\
internal error in %s, line %u"), __FUNCTION__, __LINE__);
		      }
		  }
	}
    }

  for (cnt = 0; cnt < 256; ++cnt)
    {
      uint32_t tmp = ctype->class256_collection[cnt];

      if (tmp != 0)
	{
	  for (cls1 = 0; cls1 < NCLASS; ++cls1)
	    if ((tmp & _ISbit (cls1)) != 0)
	      for (cls2 = 0; cls2 < NCLASS; ++cls2)
		if (valid_table[cls1].allow[cls2] != '-')
		  {
		    int eq = (tmp & _ISbit (cls2)) != 0;
		    switch (valid_table[cls1].allow[cls2])
		      {
		      case 'M':
			if (!eq)
			  {
			    char buf[17];

			    snprintf (buf, sizeof buf, "\\%zo", cnt);

			    record_error (0, 0, _("\
character '%s' in class `%s' must be in class `%s'"),
					  buf,
					  valid_table[cls1].name,
					  valid_table[cls2].name);
			  }
			break;

		      case 'X':
			if (eq)
			  {
			    char buf[17];

			    snprintf (buf, sizeof buf, "\\%zo", cnt);

			    record_error (0, 0, _("\
character '%s' in class `%s' must not be in class `%s'"),
					  buf,
					  valid_table[cls1].name,
					  valid_table[cls2].name);
			  }
			break;

		      case 'D':
			ctype->class256_collection[cnt] |= _ISbit (cls2);
			break;

		      default:
			record_error (5, 0, _("\
internal error in %s, line %u"), __FUNCTION__, __LINE__);
		      }
		  }
	}
    }

  /* ... and now test <SP> as a special case.  */
  space_value = 32;
  if (((cnt = BITPOS (tok_space),
	(ELEM (ctype, class_collection, , space_value)
	 & BITw (tok_space)) == 0)
       || (cnt = BITPOS (tok_blank),
	   (ELEM (ctype, class_collection, , space_value)
	    & BITw (tok_blank)) == 0)))
    {
      record_error (0, 0, _("<SP> character not in class `%s'"),
		    valid_table[cnt].name);
    }
  else if (((cnt = BITPOS (tok_punct),
	     (ELEM (ctype, class_collection, , space_value)
	      & BITw (tok_punct)) != 0)
	    || (cnt = BITPOS (tok_graph),
		(ELEM (ctype, class_collection, , space_value)
		 & BITw (tok_graph))
		!= 0)))
    {
      record_error (0, 0, _("\
<SP> character must not be in class `%s'"),
				valid_table[cnt].name);
    }
  else
    ELEM (ctype, class_collection, , space_value) |= BITw (tok_print);

  space_seq = charmap_find_value (charmap, "SP", 2);
  if (space_seq == NULL)
    space_seq = charmap_find_value (charmap, "space", 5);
  if (space_seq == NULL)
    space_seq = charmap_find_value (charmap, "U00000020", 9);
  if (space_seq == NULL || space_seq->nbytes != 1)
    {
      record_error (0, 0, _("\
character <SP> not defined in character map"));
    }
  else if (((cnt = BITPOS (tok_space),
	     (ctype->class256_collection[space_seq->bytes[0]]
	      & BIT (tok_space)) == 0)
	    || (cnt = BITPOS (tok_blank),
		(ctype->class256_collection[space_seq->bytes[0]]
		 & BIT (tok_blank)) == 0)))
    {
       record_error (0, 0, _("<SP> character not in class `%s'"),
		     valid_table[cnt].name);
    }
  else if (((cnt = BITPOS (tok_punct),
	     (ctype->class256_collection[space_seq->bytes[0]]
	      & BIT (tok_punct)) != 0)
	    || (cnt = BITPOS (tok_graph),
		(ctype->class256_collection[space_seq->bytes[0]]
		 & BIT (tok_graph)) != 0)))
    {
      record_error (0, 0, _("\
<SP> character must not be in class `%s'"),
		    valid_table[cnt].name);
    }
  else
    ctype->class256_collection[space_seq->bytes[0]] |= BIT (tok_print);

  /* Check whether all single-byte characters make to their upper/lowercase
     equivalent according to the ASCII rules.  */
  for (cnt = 'A'; cnt <= 'Z'; ++cnt)
    {
      uint32_t uppval = ctype->map256_collection[0][cnt];
      uint32_t lowval = ctype->map256_collection[1][cnt];
      uint32_t lowuppval = ctype->map256_collection[0][lowval];
      uint32_t lowlowval = ctype->map256_collection[1][lowval];

      if (uppval != cnt
	  || lowval != cnt + 0x20
	  || lowuppval != cnt
	  || lowlowval != cnt + 0x20)
	ctype->nonascii_case = 1;
    }
  for (cnt = 0; cnt < 256; ++cnt)
    if (cnt < 'A' || (cnt > 'Z' && cnt < 'a') || cnt > 'z')
      if (ctype->map256_collection[0][cnt] != cnt
	  || ctype->map256_collection[1][cnt] != cnt)
	ctype->nonascii_case = 1;

  /* Now that the tests are done make sure the name array contains all
     characters which are handled in the WIDTH section of the
     character set definition file.  */
  if (charmap->width_rules != NULL)
    for (cnt = 0; cnt < charmap->nwidth_rules; ++cnt)
      {
	unsigned char bytes[charmap->mb_cur_max];
	int nbytes = charmap->width_rules[cnt].from->nbytes;

	/* We have the range of character for which the width is
           specified described using byte sequences of the multibyte
           charset.  We have to convert this to UCS4 now.  And we
           cannot simply convert the beginning and the end of the
           sequence, we have to iterate over the byte sequence and
           convert it for every single character.  */
	memcpy (bytes, charmap->width_rules[cnt].from->bytes, nbytes);

	while (nbytes < charmap->width_rules[cnt].to->nbytes
	       || memcmp (bytes, charmap->width_rules[cnt].to->bytes,
			  nbytes) <= 0)
	  {
	    /* Find the UCS value for `bytes'.  */
	    int inner;
	    uint32_t wch;
	    struct charseq *seq
	      = charmap_find_symbol (charmap, (char *) bytes, nbytes);

	    if (seq == NULL)
	      wch = ILLEGAL_CHAR_VALUE;
	    else if (seq->ucs4 != UNINITIALIZED_CHAR_VALUE)
	      wch = seq->ucs4;
	    else
	      wch = repertoire_find_value (ctype->repertoire, seq->name,
					   strlen (seq->name));

	    if (wch != ILLEGAL_CHAR_VALUE)
	      /* We are only interested in the side-effects of the
		 `find_idx' call.  It will add appropriate entries in
		 the name array if this is necessary.  */
	      (void) find_idx (ctype, NULL, NULL, NULL, wch);

	    /* "Increment" the bytes sequence.  */
	    inner = nbytes - 1;
	    while (inner >= 0 && bytes[inner] == 0xff)
	      --inner;

	    if (inner < 0)
	      {
		/* We have to extend the byte sequence.  */
		if (nbytes >= charmap->width_rules[cnt].to->nbytes)
		  break;

		bytes[0] = 1;
		memset (&bytes[1], 0, nbytes);
		++nbytes;
	      }
	    else
	      {
		++bytes[inner];
		while (++inner < nbytes)
		  bytes[inner] = 0;
	      }
	  }
      }

  /* Now set all the other characters of the character set to the
     default width.  */
  curs = NULL;
  while (iterate_table (&charmap->char_table, &curs, &key, &len, &vdata) == 0)
    {
      struct charseq *data = (struct charseq *) vdata;

      if (data->ucs4 == UNINITIALIZED_CHAR_VALUE)
	data->ucs4 = repertoire_find_value (ctype->repertoire,
					    data->name, len);

      if (data->ucs4 != ILLEGAL_CHAR_VALUE)
	(void) find_idx (ctype, NULL, NULL, NULL, data->ucs4);
    }

  /* There must be a multiple of 10 digits.  */
  if (ctype->mbdigits_act % 10 != 0)
    {
      assert (ctype->mbdigits_act == ctype->wcdigits_act);
      ctype->wcdigits_act -= ctype->mbdigits_act % 10;
      ctype->mbdigits_act -= ctype->mbdigits_act % 10;
      record_error (0, 0, _("\
`digit' category has not entries in groups of ten"));
    }

  /* Check the input digits.  There must be a multiple of ten available.
     In each group it could be that one or the other character is missing.
     In this case the whole group must be removed.  */
  cnt = 0;
  while (cnt < ctype->mbdigits_act)
    {
      size_t inner;
      for (inner = 0; inner < 10; ++inner)
	if (ctype->mbdigits[cnt + inner] == NULL)
	  break;

      if (inner == 10)
	cnt += 10;
      else
	{
	  /* Remove the group.  */
	  memmove (&ctype->mbdigits[cnt], &ctype->mbdigits[cnt + 10],
		   ((ctype->wcdigits_act - cnt - 10)
		    * sizeof (ctype->mbdigits[0])));
	  ctype->mbdigits_act -= 10;
	}
    }

  /* If no input digits are given use the default.  */
  if (ctype->mbdigits_act == 0)
    {
      if (ctype->mbdigits_max == 0)
	{
	  ctype->mbdigits = obstack_alloc (&((struct charmap_t *) charmap)->mem_pool,
					   10 * sizeof (struct charseq *));
	  ctype->mbdigits_max = 10;
	}

      for (cnt = 0; cnt < 10; ++cnt)
	{
	  ctype->mbdigits[cnt] = charmap_find_symbol (charmap,
						      (char *) digits + cnt, 1);
	  if (ctype->mbdigits[cnt] == NULL)
	    {
	      ctype->mbdigits[cnt] = charmap_find_symbol (charmap,
							  longnames[cnt],
							  strlen (longnames[cnt]));
	      if (ctype->mbdigits[cnt] == NULL)
		{
		  /* Hum, this ain't good.  */
		  record_error (0, 0, _("\
no input digits defined and none of the standard names in the charmap"));

		  ctype->mbdigits[cnt] = obstack_alloc (&((struct charmap_t *) charmap)->mem_pool,
							sizeof (struct charseq) + 1);

		  /* This is better than nothing.  */
		  ctype->mbdigits[cnt]->bytes[0] = digits[cnt];
		  ctype->mbdigits[cnt]->nbytes = 1;
		}
	    }
	}

      ctype->mbdigits_act = 10;
    }

  /* Check the wide character input digits.  There must be a multiple
     of ten available.  In each group it could be that one or the other
     character is missing.  In this case the whole group must be
     removed.  */
  cnt = 0;
  while (cnt < ctype->wcdigits_act)
    {
      size_t inner;
      for (inner = 0; inner < 10; ++inner)
	if (ctype->wcdigits[cnt + inner] == ILLEGAL_CHAR_VALUE)
	  break;

      if (inner == 10)
	cnt += 10;
      else
	{
	  /* Remove the group.  */
	  memmove (&ctype->wcdigits[cnt], &ctype->wcdigits[cnt + 10],
		   ((ctype->wcdigits_act - cnt - 10)
		    * sizeof (ctype->wcdigits[0])));
	  ctype->wcdigits_act -= 10;
	}
    }

  /* If no input digits are given use the default.  */
  if (ctype->wcdigits_act == 0)
    {
      if (ctype->wcdigits_max == 0)
	{
	  ctype->wcdigits = obstack_alloc (&((struct charmap_t *) charmap)->mem_pool,
					   10 * sizeof (uint32_t));
	  ctype->wcdigits_max = 10;
	}

      for (cnt = 0; cnt < 10; ++cnt)
	ctype->wcdigits[cnt] = L'0' + cnt;

      ctype->mbdigits_act = 10;
    }

  /* Check the outdigits.  */
  warned = 0;
  for (cnt = 0; cnt < 10; ++cnt)
    if (ctype->mboutdigits[cnt] == NULL)
      {
	if (!warned)
	  {
	    record_error (0, 0, _("\
not all characters used in `outdigit' are available in the charmap"));
	    warned = 1;
	  }

	static const struct charseq replace =
	  {
	     .nbytes = 1,
	     .bytes = "?",
	  };
	ctype->mboutdigits[cnt] = (struct charseq *) &replace;
      }

  warned = 0;
  for (cnt = 0; cnt < 10; ++cnt)
    if (ctype->wcoutdigits[cnt] == 0)
      {
	if (!warned)
	  {
	    record_error (0, 0, _("\
not all characters used in `outdigit' are available in the repertoire"));
	    warned = 1;
	  }

	ctype->wcoutdigits[cnt] = L'?';
      }

  /* Sort the entries in the translit_ignore list.  */
  if (ctype->translit_ignore != NULL)
    {
      struct translit_ignore_t *firstp = ctype->translit_ignore;
      struct translit_ignore_t *runp;

      ctype->ntranslit_ignore = 1;

      for (runp = firstp->next; runp != NULL; runp = runp->next)
	{
	  struct translit_ignore_t *lastp = NULL;
	  struct translit_ignore_t *cmpp;

	  ++ctype->ntranslit_ignore;

	  for (cmpp = firstp; cmpp != NULL; lastp = cmpp, cmpp = cmpp->next)
	    if (runp->from < cmpp->from)
	      break;

	  runp->next = lastp;
	  if (lastp == NULL)
	    firstp = runp;
	}

      ctype->translit_ignore = firstp;
    }
}


void
ctype_output (struct localedef_t *locale, const struct charmap_t *charmap,
	      const char *output_path)
{
  struct locale_ctype_t *ctype = locale->categories[LC_CTYPE].ctype;
  const size_t nelems = (_NL_ITEM_INDEX (_NL_CTYPE_EXTRA_MAP_1)
			 + ctype->nr_charclass + ctype->map_collection_nr);
  struct locale_file file;
  uint32_t default_missing_len;
  size_t elem, cnt;

  /* Now prepare the output: Find the sizes of the table we can use.  */
  allocate_arrays (ctype, charmap, ctype->repertoire);

  default_missing_len = (ctype->default_missing
			 ? wcslen ((wchar_t *) ctype->default_missing)
			 : 0);

  init_locale_data (&file, nelems);
  for (elem = 0; elem < nelems; ++elem)
    {
      if (elem < _NL_ITEM_INDEX (_NL_CTYPE_EXTRA_MAP_1))
	switch (elem)
	  {
#define CTYPE_EMPTY(name) \
	  case name:							      \
	    add_locale_empty (&file);					      \
	    break

	  CTYPE_EMPTY(_NL_CTYPE_GAP1);
	  CTYPE_EMPTY(_NL_CTYPE_GAP2);
	  CTYPE_EMPTY(_NL_CTYPE_GAP3);
	  CTYPE_EMPTY(_NL_CTYPE_GAP4);
	  CTYPE_EMPTY(_NL_CTYPE_GAP5);
	  CTYPE_EMPTY(_NL_CTYPE_GAP6);

#define CTYPE_RAW_DATA(name, base, size)				      \
	  case _NL_ITEM_INDEX (name):					      \
	    add_locale_raw_data (&file, base, size);			      \
	    break

	  CTYPE_RAW_DATA (_NL_CTYPE_CLASS,
			  ctype->ctype_b,
			  (256 + 128) * sizeof (char_class_t));

#define CTYPE_UINT32_ARRAY(name, base, n_elems)				      \
	  case _NL_ITEM_INDEX (name):					      \
	    add_locale_uint32_array (&file, base, n_elems);		      \
	    break

	  CTYPE_UINT32_ARRAY (_NL_CTYPE_TOUPPER, ctype->map_b[0], 256 + 128);
	  CTYPE_UINT32_ARRAY (_NL_CTYPE_TOLOWER, ctype->map_b[1], 256 + 128);
	  CTYPE_UINT32_ARRAY (_NL_CTYPE_TOUPPER32, ctype->map32_b[0], 256);
	  CTYPE_UINT32_ARRAY (_NL_CTYPE_TOLOWER32, ctype->map32_b[1], 256);
	  CTYPE_RAW_DATA (_NL_CTYPE_CLASS32,
			  ctype->ctype32_b,
			  256 * sizeof (char_class32_t));

#define CTYPE_UINT32(name, value)					      \
	  case _NL_ITEM_INDEX (name):					      \
	    add_locale_uint32 (&file, value);				      \
	    break

	  CTYPE_UINT32 (_NL_CTYPE_CLASS_OFFSET, ctype->class_offset);
	  CTYPE_UINT32 (_NL_CTYPE_MAP_OFFSET, ctype->map_offset);
	  CTYPE_UINT32 (_NL_CTYPE_TRANSLIT_TAB_SIZE, ctype->translit_idx_size);

	  CTYPE_UINT32_ARRAY (_NL_CTYPE_TRANSLIT_FROM_IDX,
			      ctype->translit_from_idx,
			      ctype->translit_idx_size);

	  CTYPE_UINT32_ARRAY (_NL_CTYPE_TRANSLIT_FROM_TBL,
			      ctype->translit_from_tbl,
			      ctype->translit_from_tbl_size
			      / sizeof (uint32_t));

	  CTYPE_UINT32_ARRAY (_NL_CTYPE_TRANSLIT_TO_IDX,
			      ctype->translit_to_idx,
			      ctype->translit_idx_size);

	  CTYPE_UINT32_ARRAY (_NL_CTYPE_TRANSLIT_TO_TBL,
			      ctype->translit_to_tbl,
			      ctype->translit_to_tbl_size / sizeof (uint32_t));

	  case _NL_ITEM_INDEX (_NL_CTYPE_CLASS_NAMES):
	    /* The class name array.  */
	    start_locale_structure (&file);
	    for (cnt = 0; cnt < ctype->nr_charclass; ++cnt)
	      add_locale_string (&file, ctype->classnames[cnt]);
	    add_locale_char (&file, 0);
	    align_locale_data (&file, LOCFILE_ALIGN);
	    end_locale_structure (&file);
	    break;

	  case _NL_ITEM_INDEX (_NL_CTYPE_MAP_NAMES):
	    /* The class name array.  */
	    start_locale_structure (&file);
	    for (cnt = 0; cnt < ctype->map_collection_nr; ++cnt)
	      add_locale_string (&file, ctype->mapnames[cnt]);
	    add_locale_char (&file, 0);
	    align_locale_data (&file, LOCFILE_ALIGN);
	    end_locale_structure (&file);
	    break;

	  case _NL_ITEM_INDEX (_NL_CTYPE_WIDTH):
	    add_locale_wcwidth_table (&file, &ctype->width);
	    break;

	  CTYPE_UINT32 (_NL_CTYPE_MB_CUR_MAX, ctype->mb_cur_max);

	  case _NL_ITEM_INDEX (_NL_CTYPE_CODESET_NAME):
	    add_locale_string (&file, ctype->codeset_name);
	    break;

	  CTYPE_UINT32 (_NL_CTYPE_MAP_TO_NONASCII, ctype->to_nonascii);

	  CTYPE_UINT32 (_NL_CTYPE_NONASCII_CASE, ctype->nonascii_case);

	  case _NL_ITEM_INDEX (_NL_CTYPE_INDIGITS_MB_LEN):
	    add_locale_uint32 (&file, ctype->mbdigits_act / 10);
	    break;

	  case _NL_ITEM_INDEX (_NL_CTYPE_INDIGITS_WC_LEN):
	    add_locale_uint32 (&file, ctype->wcdigits_act / 10);
	    break;

	  case _NL_ITEM_INDEX (_NL_CTYPE_INDIGITS0_MB) ... _NL_ITEM_INDEX (_NL_CTYPE_INDIGITS9_MB):
	    start_locale_structure (&file);
	    for (cnt = elem - _NL_ITEM_INDEX (_NL_CTYPE_INDIGITS0_MB);
		 cnt < ctype->mbdigits_act; cnt += 10)
	      {
		add_locale_raw_data (&file, ctype->mbdigits[cnt]->bytes,
				     ctype->mbdigits[cnt]->nbytes);
		add_locale_char (&file, 0);
	      }
	    end_locale_structure (&file);
	    break;

	  case _NL_ITEM_INDEX (_NL_CTYPE_OUTDIGIT0_MB) ... _NL_ITEM_INDEX (_NL_CTYPE_OUTDIGIT9_MB):
	    start_locale_structure (&file);
	    cnt = elem - _NL_ITEM_INDEX (_NL_CTYPE_OUTDIGIT0_MB);
	    add_locale_raw_data (&file, ctype->mboutdigits[cnt]->bytes,
				 ctype->mboutdigits[cnt]->nbytes);
	    add_locale_char (&file, 0);
	    end_locale_structure (&file);
	    break;

	  case _NL_ITEM_INDEX (_NL_CTYPE_INDIGITS0_WC) ... _NL_ITEM_INDEX (_NL_CTYPE_INDIGITS9_WC):
	    start_locale_structure (&file);
	    for (cnt = elem - _NL_ITEM_INDEX (_NL_CTYPE_INDIGITS0_WC);
		 cnt < ctype->wcdigits_act; cnt += 10)
	      add_locale_uint32 (&file, ctype->wcdigits[cnt]);
	    end_locale_structure (&file);
	    break;

	  case _NL_ITEM_INDEX (_NL_CTYPE_OUTDIGIT0_WC) ... _NL_ITEM_INDEX (_NL_CTYPE_OUTDIGIT9_WC):
	    cnt = elem - _NL_ITEM_INDEX (_NL_CTYPE_OUTDIGIT0_WC);
	    add_locale_uint32 (&file, ctype->wcoutdigits[cnt]);
	    break;

	  case _NL_ITEM_INDEX(_NL_CTYPE_TRANSLIT_DEFAULT_MISSING_LEN):
	    add_locale_uint32 (&file, default_missing_len);
	    break;

	  case _NL_ITEM_INDEX(_NL_CTYPE_TRANSLIT_DEFAULT_MISSING):
	    add_locale_uint32_array (&file, ctype->default_missing,
				     default_missing_len);
	    break;

	  case _NL_ITEM_INDEX(_NL_CTYPE_TRANSLIT_IGNORE_LEN):
	    add_locale_uint32 (&file, ctype->ntranslit_ignore);
	    break;

	  case _NL_ITEM_INDEX(_NL_CTYPE_TRANSLIT_IGNORE):
	    start_locale_structure (&file);
	    {
	      struct translit_ignore_t *runp;
	      for (runp = ctype->translit_ignore; runp != NULL;
		   runp = runp->next)
		{
		  add_locale_uint32 (&file, runp->from);
		  add_locale_uint32 (&file, runp->to);
		  add_locale_uint32 (&file, runp->step);
		}
	    }
	    end_locale_structure (&file);
	    break;

	  default:
	    assert (! "unknown CTYPE element");
	  }
      else
	{
	  /* Handle extra maps.  */
	  size_t nr = elem - _NL_ITEM_INDEX (_NL_CTYPE_EXTRA_MAP_1);
	  if (nr < ctype->nr_charclass)
	    {
	      start_locale_prelude (&file);
	      add_locale_uint32_array (&file, ctype->class_b[nr], 256 / 32);
	      end_locale_prelude (&file);
	      add_locale_wctype_table (&file, &ctype->class_3level[nr]);
	    }
	  else
	    {
	      nr -= ctype->nr_charclass;
	      assert (nr < ctype->map_collection_nr);
	      add_locale_wctrans_table (&file, &ctype->map_3level[nr]);
	    }
	}
    }

  write_locale_data (output_path, LC_CTYPE, "LC_CTYPE", &file);
}


/* Local functions.  */
static void
ctype_class_new (struct linereader *lr, struct locale_ctype_t *ctype,
		 const char *name)
{
  size_t cnt;

  for (cnt = 0; cnt < ctype->nr_charclass; ++cnt)
    if (strcmp (ctype->classnames[cnt], name) == 0)
      break;

  if (cnt < ctype->nr_charclass)
    {
      lr_error (lr, _("character class `%s' already defined"), name);
      return;
    }

  if (ctype->nr_charclass == MAX_NR_CHARCLASS)
    /* Exit code 2 is prescribed in P1003.2b.  */
    record_error (2, 0, _("\
implementation limit: no more than %Zd character classes allowed"),
		  MAX_NR_CHARCLASS);

  ctype->classnames[ctype->nr_charclass++] = name;
}


static void
ctype_map_new (struct linereader *lr, struct locale_ctype_t *ctype,
	       const char *name, const struct charmap_t *charmap)
{
  size_t max_chars = 0;
  size_t cnt;

  for (cnt = 0; cnt < ctype->map_collection_nr; ++cnt)
    {
      if (strcmp (ctype->mapnames[cnt], name) == 0)
	break;

      if (max_chars < ctype->map_collection_max[cnt])
	max_chars = ctype->map_collection_max[cnt];
    }

  if (cnt < ctype->map_collection_nr)
    {
      lr_error (lr, _("character map `%s' already defined"), name);
      return;
    }

  if (ctype->map_collection_nr == MAX_NR_CHARMAP)
    /* Exit code 2 is prescribed in P1003.2b.  */
    record_error (2, 0, _("\
implementation limit: no more than %d character maps allowed"),
		  MAX_NR_CHARMAP);

  ctype->mapnames[cnt] = name;

  if (max_chars == 0)
    ctype->map_collection_max[cnt] = charmap->mb_cur_max == 1 ? 256 : 512;
  else
    ctype->map_collection_max[cnt] = max_chars;

  ctype->map_collection[cnt] = (uint32_t *)
    xcalloc (sizeof (uint32_t), ctype->map_collection_max[cnt]);
  ctype->map_collection_act[cnt] = 256;

  ++ctype->map_collection_nr;
}


/* We have to be prepared that TABLE, MAX, and ACT can be NULL.  This
   is possible if we only want to extend the name array.  */
static uint32_t *
find_idx (struct locale_ctype_t *ctype, uint32_t **table, size_t *max,
	  size_t *act, uint32_t idx)
{
  size_t cnt;

  if (idx < 256)
    return table == NULL ? NULL : &(*table)[idx];

  /* Use the charnames_idx lookup table instead of the slow search loop.  */
#if 1
  cnt = idx_table_get (&ctype->charnames_idx, idx);
  if (cnt == EMPTY)
    /* Not found.  */
    cnt = ctype->charnames_act;
#else
  for (cnt = 256; cnt < ctype->charnames_act; ++cnt)
    if (ctype->charnames[cnt] == idx)
      break;
#endif

  /* We have to distinguish two cases: the name is found or not.  */
  if (cnt == ctype->charnames_act)
    {
      /* Extend the name array.  */
      if (ctype->charnames_act == ctype->charnames_max)
	{
	  ctype->charnames_max *= 2;
	  ctype->charnames = (uint32_t *)
	    xrealloc (ctype->charnames,
		      sizeof (uint32_t) * ctype->charnames_max);
	}
      ctype->charnames[ctype->charnames_act++] = idx;
      idx_table_add (&ctype->charnames_idx, idx, cnt);
    }

  if (table == NULL)
    /* We have done everything we are asked to do.  */
    return NULL;

  if (max == NULL)
    /* The caller does not want to extend the table.  */
    return (cnt >= *act ? NULL : &(*table)[cnt]);

  if (cnt >= *act)
    {
      if (cnt >= *max)
	{
	  size_t old_max = *max;
	  do
	    *max *= 2;
	  while (*max <= cnt);

	  *table =
	    (uint32_t *) xrealloc (*table, *max * sizeof (uint32_t));
	  memset (&(*table)[old_max], '\0',
		  (*max - old_max) * sizeof (uint32_t));
	}

      *act = cnt + 1;
    }

  return &(*table)[cnt];
}


static int
get_character (struct token *now, const struct charmap_t *charmap,
	       struct repertoire_t *repertoire,
	       struct charseq **seqp, uint32_t *wchp)
{
  if (now->tok == tok_bsymbol)
    {
      /* This will hopefully be the normal case.  */
      *wchp = repertoire_find_value (repertoire, now->val.str.startmb,
				     now->val.str.lenmb);
      *seqp = charmap_find_value (charmap, now->val.str.startmb,
				  now->val.str.lenmb);
    }
  else if (now->tok == tok_ucs4)
    {
      char utmp[10];

      snprintf (utmp, sizeof (utmp), "U%08X", now->val.ucs4);
      *seqp = charmap_find_value (charmap, utmp, 9);

      if (*seqp == NULL)
	*seqp = repertoire_find_seq (repertoire, now->val.ucs4);

      if (*seqp == NULL)
	{
	  /* Compute the value in the charmap from the UCS value.  */
	  const char *symbol = repertoire_find_symbol (repertoire,
						       now->val.ucs4);

	  if (symbol == NULL)
	    *seqp = NULL;
	  else
	    *seqp = charmap_find_value (charmap, symbol, strlen (symbol));

	  if (*seqp == NULL)
	    {
	      if (repertoire != NULL)
		{
		  /* Insert a negative entry.  */
		  static const struct charseq negative
		    = { .ucs4 = ILLEGAL_CHAR_VALUE };
		  uint32_t *newp = obstack_alloc (&repertoire->mem_pool,
						  sizeof (uint32_t));
		  *newp = now->val.ucs4;

		  insert_entry (&repertoire->seq_table, newp,
				sizeof (uint32_t), (void *) &negative);
		}
	    }
	  else
	    (*seqp)->ucs4 = now->val.ucs4;
	}
      else if ((*seqp)->ucs4 != now->val.ucs4)
	*seqp = NULL;

      *wchp = now->val.ucs4;
    }
  else if (now->tok == tok_charcode)
    {
      /* We must map from the byte code to UCS4.  */
      *seqp = charmap_find_symbol (charmap, now->val.str.startmb,
				   now->val.str.lenmb);

      if (*seqp == NULL)
	*wchp = ILLEGAL_CHAR_VALUE;
      else
	{
	  if ((*seqp)->ucs4 == UNINITIALIZED_CHAR_VALUE)
	    (*seqp)->ucs4 = repertoire_find_value (repertoire, (*seqp)->name,
						   strlen ((*seqp)->name));
	  *wchp = (*seqp)->ucs4;
	}
    }
  else
    return 1;

  return 0;
}


/* Ellipsis like in `<foo123>..<foo12a>' or `<j1234>....<j1245>' and
   the .(2). counterparts.  */
static void
charclass_symbolic_ellipsis (struct linereader *ldfile,
			     struct locale_ctype_t *ctype,
			     const struct charmap_t *charmap,
			     struct repertoire_t *repertoire,
			     struct token *now,
			     const char *last_str,
			     unsigned long int class256_bit,
			     unsigned long int class_bit, int base,
			     int ignore_content, int handle_digits, int step)
{
  const char *nowstr = now->val.str.startmb;
  char tmp[now->val.str.lenmb + 1];
  const char *cp;
  char *endp;
  unsigned long int from;
  unsigned long int to;

  /* We have to compute the ellipsis values using the symbolic names.  */
  assert (last_str != NULL);

  if (strlen (last_str) != now->val.str.lenmb)
    {
    invalid_range:
      lr_error (ldfile,
		_("`%s' and `%.*s' are not valid names for symbolic range"),
		last_str, (int) now->val.str.lenmb, nowstr);
      return;
    }

  if (memcmp (last_str, nowstr, now->val.str.lenmb) == 0)
    /* Nothing to do, the names are the same.  */
    return;

  for (cp = last_str; *cp == *(nowstr + (cp - last_str)); ++cp)
    ;

  errno = 0;
  from = strtoul (cp, &endp, base);
  if ((from == UINT_MAX && errno == ERANGE) || *endp != '\0')
    goto invalid_range;

  to = strtoul (nowstr + (cp - last_str), &endp, base);
  if ((to == UINT_MAX && errno == ERANGE)
      || (endp - nowstr) != now->val.str.lenmb || from >= to)
    goto invalid_range;

  /* OK, we have a range FROM - TO.  Now we can create the symbolic names.  */
  if (!ignore_content)
    {
      now->val.str.startmb = tmp;
      while ((from += step) <= to)
	{
	  struct charseq *seq;
	  uint32_t wch;

	  sprintf (tmp, (base == 10 ? "%.*s%0*ld" : "%.*s%0*lX"),
		   (int) (cp - last_str), last_str,
		   (int) (now->val.str.lenmb - (cp - last_str)),
		   from);

	  if (get_character (now, charmap, repertoire, &seq, &wch))
	    goto invalid_range;

	  if (seq != NULL && seq->nbytes == 1)
	    /* Yep, we can store information about this byte sequence.  */
	    ctype->class256_collection[seq->bytes[0]] |= class256_bit;

	  if (wch != ILLEGAL_CHAR_VALUE && class_bit != 0)
	    /* We have the UCS4 position.  */
	    *find_idx (ctype, &ctype->class_collection,
		       &ctype->class_collection_max,
		       &ctype->class_collection_act, wch) |= class_bit;

	  if (handle_digits == 1)
	    {
	      /* We must store the digit values.  */
	      if (ctype->mbdigits_act == ctype->mbdigits_max)
		{
		  ctype->mbdigits_max *= 2;
		  ctype->mbdigits = xrealloc (ctype->mbdigits,
					      (ctype->mbdigits_max
					       * sizeof (char *)));
		  ctype->wcdigits_max *= 2;
		  ctype->wcdigits = xrealloc (ctype->wcdigits,
					      (ctype->wcdigits_max
					       * sizeof (uint32_t)));
		}

	      ctype->mbdigits[ctype->mbdigits_act++] = seq;
	      ctype->wcdigits[ctype->wcdigits_act++] = wch;
	    }
	  else if (handle_digits == 2)
	    {
	      /* We must store the digit values.  */
	      if (ctype->outdigits_act >= 10)
		{
		  lr_error (ldfile, _("\
%s: field `%s' does not contain exactly ten entries"),
			    "LC_CTYPE", "outdigit");
		  return;
		}

	      ctype->mboutdigits[ctype->outdigits_act] = seq;
	      ctype->wcoutdigits[ctype->outdigits_act] = wch;
	      ++ctype->outdigits_act;
	    }
	}
    }
}


/* Ellipsis like in `<U1234>..<U2345>' or `<U1234>..(2)..<U2345>'.  */
static void
charclass_ucs4_ellipsis (struct linereader *ldfile,
			 struct locale_ctype_t *ctype,
			 const struct charmap_t *charmap,
			 struct repertoire_t *repertoire,
			 struct token *now, uint32_t last_wch,
			 unsigned long int class256_bit,
			 unsigned long int class_bit, int ignore_content,
			 int handle_digits, int step)
{
  if (last_wch > now->val.ucs4)
    {
      lr_error (ldfile, _("\
to-value <U%0*X> of range is smaller than from-value <U%0*X>"),
		(now->val.ucs4 | last_wch) < 65536 ? 4 : 8, now->val.ucs4,
		(now->val.ucs4 | last_wch) < 65536 ? 4 : 8, last_wch);
      return;
    }

  if (!ignore_content)
    while ((last_wch += step) <= now->val.ucs4)
      {
	/* We have to find out whether there is a byte sequence corresponding
	   to this UCS4 value.  */
	struct charseq *seq;
	char utmp[10];

	snprintf (utmp, sizeof (utmp), "U%08X", last_wch);
	seq = charmap_find_value (charmap, utmp, 9);
	if (seq == NULL)
	  {
	    snprintf (utmp, sizeof (utmp), "U%04X", last_wch);
	    seq = charmap_find_value (charmap, utmp, 5);
	  }

	if (seq == NULL)
	  /* Try looking in the repertoire map.  */
	  seq = repertoire_find_seq (repertoire, last_wch);

	/* If this is the first time we look for this sequence create a new
	   entry.  */
	if (seq == NULL)
	  {
	    static const struct charseq negative
	      = { .ucs4 = ILLEGAL_CHAR_VALUE };

	    /* Find the symbolic name for this UCS4 value.  */
	    if (repertoire != NULL)
	      {
		const char *symbol = repertoire_find_symbol (repertoire,
							     last_wch);
		uint32_t *newp = obstack_alloc (&repertoire->mem_pool,
						sizeof (uint32_t));
		*newp = last_wch;

		if (symbol != NULL)
		  /* We have a name, now search the multibyte value.  */
		  seq = charmap_find_value (charmap, symbol, strlen (symbol));

		if (seq == NULL)
		  /* We have to create a fake entry.  */
		  seq = (struct charseq *) &negative;
		else
		  seq->ucs4 = last_wch;

		insert_entry (&repertoire->seq_table, newp, sizeof (uint32_t),
			      seq);
	      }
	    else
	      /* We have to create a fake entry.  */
	      seq = (struct charseq *) &negative;
	  }

	/* We have a name, now search the multibyte value.  */
	if (seq->ucs4 == last_wch && seq->nbytes == 1)
	  /* Yep, we can store information about this byte sequence.  */
	  ctype->class256_collection[(size_t) seq->bytes[0]]
	    |= class256_bit;

	/* And of course we have the UCS4 position.  */
	if (class_bit != 0)
	  *find_idx (ctype, &ctype->class_collection,
		     &ctype->class_collection_max,
		     &ctype->class_collection_act, last_wch) |= class_bit;

	if (handle_digits == 1)
	  {
	    /* We must store the digit values.  */
	    if (ctype->mbdigits_act == ctype->mbdigits_max)
	      {
		ctype->mbdigits_max *= 2;
		ctype->mbdigits = xrealloc (ctype->mbdigits,
					    (ctype->mbdigits_max
					     * sizeof (char *)));
		ctype->wcdigits_max *= 2;
		ctype->wcdigits = xrealloc (ctype->wcdigits,
					    (ctype->wcdigits_max
					     * sizeof (uint32_t)));
	      }

	    ctype->mbdigits[ctype->mbdigits_act++] = (seq->ucs4 == last_wch
						      ? seq : NULL);
	    ctype->wcdigits[ctype->wcdigits_act++] = last_wch;
	  }
	else if (handle_digits == 2)
	  {
	    /* We must store the digit values.  */
	    if (ctype->outdigits_act >= 10)
	      {
		lr_error (ldfile, _("\
%s: field `%s' does not contain exactly ten entries"),
			  "LC_CTYPE", "outdigit");
		return;
	      }

	    ctype->mboutdigits[ctype->outdigits_act] = (seq->ucs4 == last_wch
							? seq : NULL);
	    ctype->wcoutdigits[ctype->outdigits_act] = last_wch;
	    ++ctype->outdigits_act;
	  }
      }
}


/* Ellipsis as in `/xea/x12.../xea/x34'.  */
static void
charclass_charcode_ellipsis (struct linereader *ldfile,
			     struct locale_ctype_t *ctype,
			     const struct charmap_t *charmap,
			     struct repertoire_t *repertoire,
			     struct token *now, char *last_charcode,
			     uint32_t last_charcode_len,
			     unsigned long int class256_bit,
			     unsigned long int class_bit, int ignore_content,
			     int handle_digits)
{
  /* First check whether the to-value is larger.  */
  if (now->val.charcode.nbytes != last_charcode_len)
    {
      lr_error (ldfile, _("\
start and end character sequence of range must have the same length"));
      return;
    }

  if (memcmp (last_charcode, now->val.charcode.bytes, last_charcode_len) > 0)
    {
      lr_error (ldfile, _("\
to-value character sequence is smaller than from-value sequence"));
      return;
    }

  if (!ignore_content)
    {
      do
	{
	  /* Increment the byte sequence value.  */
	  struct charseq *seq;
	  uint32_t wch;
	  int i;

	  for (i = last_charcode_len - 1; i >= 0; --i)
	    if (++last_charcode[i] != 0)
	      break;

	  if (last_charcode_len == 1)
	    /* Of course we have the charcode value.  */
	    ctype->class256_collection[(size_t) last_charcode[0]]
	      |= class256_bit;

	  /* Find the symbolic name.  */
	  seq = charmap_find_symbol (charmap, last_charcode,
				     last_charcode_len);
	  if (seq != NULL)
	    {
	      if (seq->ucs4 == UNINITIALIZED_CHAR_VALUE)
		seq->ucs4 = repertoire_find_value (repertoire, seq->name,
						   strlen (seq->name));
	      wch = seq == NULL ? ILLEGAL_CHAR_VALUE : seq->ucs4;

	      if (wch != ILLEGAL_CHAR_VALUE && class_bit != 0)
		*find_idx (ctype, &ctype->class_collection,
			   &ctype->class_collection_max,
			   &ctype->class_collection_act, wch) |= class_bit;
	    }
	  else
	    wch = ILLEGAL_CHAR_VALUE;

	  if (handle_digits == 1)
	    {
	      /* We must store the digit values.  */
	      if (ctype->mbdigits_act == ctype->mbdigits_max)
		{
		  ctype->mbdigits_max *= 2;
		  ctype->mbdigits = xrealloc (ctype->mbdigits,
					      (ctype->mbdigits_max
					       * sizeof (char *)));
		  ctype->wcdigits_max *= 2;
		  ctype->wcdigits = xrealloc (ctype->wcdigits,
					      (ctype->wcdigits_max
					       * sizeof (uint32_t)));
		}

	      seq = xmalloc (sizeof (struct charseq) + last_charcode_len);
	      memcpy ((char *) (seq + 1), last_charcode, last_charcode_len);
	      seq->nbytes = last_charcode_len;

	      ctype->mbdigits[ctype->mbdigits_act++] = seq;
	      ctype->wcdigits[ctype->wcdigits_act++] = wch;
	    }
	  else if (handle_digits == 2)
	    {
	      struct charseq *seq;
	      /* We must store the digit values.  */
	      if (ctype->outdigits_act >= 10)
		{
		  lr_error (ldfile, _("\
%s: field `%s' does not contain exactly ten entries"),
			    "LC_CTYPE", "outdigit");
		  return;
		}

	      seq = xmalloc (sizeof (struct charseq) + last_charcode_len);
	      memcpy ((char *) (seq + 1), last_charcode, last_charcode_len);
	      seq->nbytes = last_charcode_len;

	      ctype->mboutdigits[ctype->outdigits_act] = seq;
	      ctype->wcoutdigits[ctype->outdigits_act] = wch;
	      ++ctype->outdigits_act;
	    }
	}
      while (memcmp (last_charcode, now->val.charcode.bytes,
		     last_charcode_len) != 0);
    }
}


static uint32_t *
find_translit2 (struct locale_ctype_t *ctype, const struct charmap_t *charmap,
		uint32_t wch)
{
  struct translit_t *trunp = ctype->translit;
  struct translit_ignore_t *tirunp = ctype->translit_ignore;

  while (trunp != NULL)
    {
      /* XXX We simplify things here.  The transliterations we look
	 for are only allowed to have one character.  */
      if (trunp->from[0] == wch && trunp->from[1] == 0)
	{
	  /* Found it.  Now look for a transliteration which can be
	     represented with the character set.  */
	  struct translit_to_t *torunp = trunp->to;

	  while (torunp != NULL)
	    {
	      int i;

	      for (i = 0; torunp->str[i] != 0; ++i)
		{
		  char utmp[10];

		  snprintf (utmp, sizeof (utmp), "U%08X", torunp->str[i]);
		  if (charmap_find_value (charmap, utmp, 9) == NULL)
		    /* This character cannot be represented.  */
		    break;
		}

	      if (torunp->str[i] == 0)
		return torunp->str;

	      torunp = torunp->next;
	    }

	  break;
	}

      trunp = trunp->next;
    }

  /* Check for ignored chars.  */
  while (tirunp != NULL)
    {
      if (tirunp->from <= wch && tirunp->to >= wch)
	{
	  uint32_t wi;

	  for (wi = tirunp->from; wi <= wch; wi += tirunp->step)
	    if (wi == wch)
	      return no_str;
	}
    }

  /* Nothing found.  */
  return NULL;
}


uint32_t *
find_translit (struct localedef_t *locale, const struct charmap_t *charmap,
	       uint32_t wch)
{
  struct locale_ctype_t *ctype;
  uint32_t *result = NULL;

  assert (locale != NULL);
  ctype = locale->categories[LC_CTYPE].ctype;

  if (ctype == NULL)
    return NULL;

  if (ctype->translit != NULL)
    result = find_translit2 (ctype, charmap, wch);

  if (result == NULL)
    {
      struct translit_include_t *irunp = ctype->translit_include;

      while (irunp != NULL && result == NULL)
	{
	  result = find_translit (find_locale (CTYPE_LOCALE,
					       irunp->copy_locale,
					       irunp->copy_repertoire,
					       charmap),
				  charmap, wch);
	  irunp = irunp->next;
	}
    }

  return result;
}


/* Read one transliteration entry.  */
static uint32_t *
read_widestring (struct linereader *ldfile, struct token *now,
		 const struct charmap_t *charmap,
		 struct repertoire_t *repertoire)
{
  uint32_t *wstr;

  if (now->tok == tok_default_missing)
    /* The special name "" will denote this case.  */
    wstr = no_str;
  else if (now->tok == tok_bsymbol)
    {
      /* Get the value from the repertoire.  */
      wstr = (uint32_t *) xmalloc (2 * sizeof (uint32_t));
      wstr[0] = repertoire_find_value (repertoire, now->val.str.startmb,
				       now->val.str.lenmb);
      if (wstr[0] == ILLEGAL_CHAR_VALUE)
	{
	  /* We cannot proceed, we don't know the UCS4 value.  */
	  free (wstr);
	  return NULL;
	}

      wstr[1] = 0;
    }
  else if (now->tok == tok_ucs4)
    {
      wstr = (uint32_t *) xmalloc (2 * sizeof (uint32_t));
      wstr[0] = now->val.ucs4;
      wstr[1] = 0;
    }
  else if (now->tok == tok_charcode)
    {
      /* Argh, we have to convert to the symbol name first and then to the
	 UCS4 value.  */
      struct charseq *seq = charmap_find_symbol (charmap,
						 now->val.str.startmb,
						 now->val.str.lenmb);
      if (seq == NULL)
	/* Cannot find the UCS4 value.  */
	return NULL;

      if (seq->ucs4 == UNINITIALIZED_CHAR_VALUE)
	seq->ucs4 = repertoire_find_value (repertoire, seq->name,
					   strlen (seq->name));
      if (seq->ucs4 == ILLEGAL_CHAR_VALUE)
	/* We cannot proceed, we don't know the UCS4 value.  */
	return NULL;

      wstr = (uint32_t *) xmalloc (2 * sizeof (uint32_t));
      wstr[0] = seq->ucs4;
      wstr[1] = 0;
    }
  else if (now->tok == tok_string)
    {
      wstr = now->val.str.startwc;
      if (wstr == NULL || wstr[0] == 0)
	return NULL;
    }
  else
    {
      if (now->tok != tok_eol && now->tok != tok_eof)
	lr_ignore_rest (ldfile, 0);
      SYNTAX_ERROR (_("%s: syntax error"), "LC_CTYPE");
      return (uint32_t *) -1l;
    }

  return wstr;
}


static void
read_translit_entry (struct linereader *ldfile, struct locale_ctype_t *ctype,
		     struct token *now, const struct charmap_t *charmap,
		     struct repertoire_t *repertoire)
{
  uint32_t *from_wstr = read_widestring (ldfile, now, charmap, repertoire);
  struct translit_t *result;
  struct translit_to_t **top;
  struct obstack *ob = &ctype->mempool;
  int first;
  int ignore;

  if (from_wstr == NULL)
    /* There is no valid from string.  */
    return;

  result = (struct translit_t *) obstack_alloc (ob,
						sizeof (struct translit_t));
  result->from = from_wstr;
  result->fname = ldfile->fname;
  result->lineno = ldfile->lineno;
  result->next = NULL;
  result->to = NULL;
  top = &result->to;
  first = 1;
  ignore = 0;

  while (1)
    {
      uint32_t *to_wstr;

      /* Next we have one or more transliterations.  They are
	 separated by semicolons.  */
      now = lr_token (ldfile, charmap, NULL, repertoire, verbose);

      if (!first && (now->tok == tok_semicolon || now->tok == tok_eol))
	{
	  /* One string read.  */
	  const uint32_t zero = 0;

	  if (!ignore)
	    {
	      obstack_grow (ob, &zero, 4);
	      to_wstr = obstack_finish (ob);

	      *top = obstack_alloc (ob, sizeof (struct translit_to_t));
	      (*top)->str = to_wstr;
	      (*top)->next = NULL;
	    }

	  if (now->tok == tok_eol)
	    {
	      result->next = ctype->translit;
	      ctype->translit = result;
	      return;
	    }

	  if (!ignore)
	    top = &(*top)->next;
	  ignore = 0;
	}
      else
	{
	  to_wstr = read_widestring (ldfile, now, charmap, repertoire);
	  if (to_wstr == (uint32_t *) -1l)
	    {
	      /* An error occurred.  */
	      obstack_free (ob, result);
	      return;
	    }

	  if (to_wstr == NULL)
	    ignore = 1;
	  else
	    /* This value is usable.  */
	    obstack_grow (ob, to_wstr, wcslen ((wchar_t *) to_wstr) * 4);

	  first = 0;
	}
    }
}


static void
read_translit_ignore_entry (struct linereader *ldfile,
			    struct locale_ctype_t *ctype,
			    const struct charmap_t *charmap,
			    struct repertoire_t *repertoire)
{
  /* We expect a semicolon-separated list of characters we ignore.  We are
     only interested in the wide character definitions.  These must be
     single characters, possibly defining a range when an ellipsis is used.  */
  while (1)
    {
      struct token *now = lr_token (ldfile, charmap, NULL, repertoire,
				    verbose);
      struct translit_ignore_t *newp;
      uint32_t from;

      if (now->tok == tok_eol || now->tok == tok_eof)
	{
	  lr_error (ldfile,
		    _("premature end of `translit_ignore' definition"));
	  return;
	}

      if (now->tok != tok_bsymbol && now->tok != tok_ucs4)
	{
	  lr_error (ldfile, _("syntax error"));
	  lr_ignore_rest (ldfile, 0);
	  return;
	}

      if (now->tok == tok_ucs4)
	from = now->val.ucs4;
      else
	/* Try to get the value.  */
	from = repertoire_find_value (repertoire, now->val.str.startmb,
				      now->val.str.lenmb);

      if (from == ILLEGAL_CHAR_VALUE)
	{
	  lr_error (ldfile, "invalid character name");
	  newp = NULL;
	}
      else
	{
	  newp = (struct translit_ignore_t *)
	    obstack_alloc (&ctype->mempool, sizeof (struct translit_ignore_t));
	  newp->from = from;
	  newp->to = from;
	  newp->step = 1;

	  newp->next = ctype->translit_ignore;
	  ctype->translit_ignore = newp;
	}

      /* Now we expect either a semicolon, an ellipsis, or the end of the
	 line.  */
      now = lr_token (ldfile, charmap, NULL, repertoire, verbose);

      if (now->tok == tok_ellipsis2 || now->tok == tok_ellipsis2_2)
	{
	  /* XXX Should we bother implementing `....'?  `...' certainly
	     will not be implemented.  */
	  uint32_t to;
	  int step = now->tok == tok_ellipsis2_2 ? 2 : 1;

	  now = lr_token (ldfile, charmap, NULL, repertoire, verbose);

	  if (now->tok == tok_eol || now->tok == tok_eof)
	    {
	      lr_error (ldfile,
			_("premature end of `translit_ignore' definition"));
	      return;
	    }

	  if (now->tok != tok_bsymbol && now->tok != tok_ucs4)
	    {
	      lr_error (ldfile, _("syntax error"));
	      lr_ignore_rest (ldfile, 0);
	      return;
	    }

	  if (now->tok == tok_ucs4)
	    to = now->val.ucs4;
	  else
	    /* Try to get the value.  */
	    to = repertoire_find_value (repertoire, now->val.str.startmb,
					now->val.str.lenmb);

	  if (to == ILLEGAL_CHAR_VALUE)
	    lr_error (ldfile, "invalid character name");
	  else
	    {
	      /* Make sure the `to'-value is larger.  */
	      if (to >= from)
		{
		  newp->to = to;
		  newp->step = step;
		}
	      else
		lr_error (ldfile, _("\
to-value <U%0*X> of range is smaller than from-value <U%0*X>"),
			  (to | from) < 65536 ? 4 : 8, to,
			  (to | from) < 65536 ? 4 : 8, from);
	    }

	  /* And the next token.  */
	  now = lr_token (ldfile, charmap, NULL, repertoire, verbose);
	}

      if (now->tok == tok_eol || now->tok == tok_eof)
	/* We are done.  */
	return;

      if (now->tok == tok_semicolon)
	/* Next round.  */
	continue;

      /* If we come here something is wrong.  */
      lr_error (ldfile, _("syntax error"));
      lr_ignore_rest (ldfile, 0);
      return;
    }
}


/* The parser for the LC_CTYPE section of the locale definition.  */
void
ctype_read (struct linereader *ldfile, struct localedef_t *result,
	    const struct charmap_t *charmap, const char *repertoire_name,
	    int ignore_content)
{
  struct repertoire_t *repertoire = NULL;
  struct locale_ctype_t *ctype;
  struct token *now;
  enum token_t nowtok;
  size_t cnt;
  uint32_t last_wch = 0;
  enum token_t last_token;
  enum token_t ellipsis_token;
  int step;
  char last_charcode[16];
  size_t last_charcode_len = 0;
  const char *last_str = NULL;
  int mapidx;
  struct localedef_t *copy_locale = NULL;

  /* Get the repertoire we have to use.  */
  if (repertoire_name != NULL)
    repertoire = repertoire_read (repertoire_name);

  /* The rest of the line containing `LC_CTYPE' must be free.  */
  lr_ignore_rest (ldfile, 1);


  do
    {
      now = lr_token (ldfile, charmap, NULL, NULL, verbose);
      nowtok = now->tok;
    }
  while (nowtok == tok_eol);

  /* If we see `copy' now we are almost done.  */
  if (nowtok == tok_copy)
    {
      now = lr_token (ldfile, charmap, NULL, NULL, verbose);
      if (now->tok != tok_string)
	{
	  SYNTAX_ERROR (_("%s: syntax error"), "LC_CTYPE");

	skip_category:
	  do
	    now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	  while (now->tok != tok_eof && now->tok != tok_end);

	  if (now->tok != tok_eof
	      || (now = lr_token (ldfile, charmap, NULL, NULL, verbose),
		  now->tok == tok_eof))
	    lr_error (ldfile, _("%s: premature end of file"), "LC_CTYPE");
	  else if (now->tok != tok_lc_ctype)
	    {
	      lr_error (ldfile, _("\
%1$s: definition does not end with `END %1$s'"), "LC_CTYPE");
	      lr_ignore_rest (ldfile, 0);
	    }
	  else
	    lr_ignore_rest (ldfile, 1);

	  return;
	}

      if (! ignore_content)
	{
	  /* Get the locale definition.  */
	  copy_locale = load_locale (LC_CTYPE, now->val.str.startmb,
				     repertoire_name, charmap, NULL);
	  if ((copy_locale->avail & CTYPE_LOCALE) == 0)
	    {
	      /* Not yet loaded.  So do it now.  */
	      if (locfile_read (copy_locale, charmap) != 0)
		goto skip_category;
	    }

	  if (copy_locale->categories[LC_CTYPE].ctype == NULL)
	    return;
	}

      lr_ignore_rest (ldfile, 1);

      now = lr_token (ldfile, charmap, NULL, NULL, verbose);
      nowtok = now->tok;
    }

  /* Prepare the data structures.  */
  ctype_startup (ldfile, result, charmap, copy_locale, ignore_content);
  ctype = result->categories[LC_CTYPE].ctype;

  /* Remember the repertoire we use.  */
  if (!ignore_content)
    ctype->repertoire = repertoire;

  while (1)
    {
      unsigned long int class_bit = 0;
      unsigned long int class256_bit = 0;
      int handle_digits = 0;

      /* Of course we don't proceed beyond the end of file.  */
      if (nowtok == tok_eof)
	break;

      /* Ingore empty lines.  */
      if (nowtok == tok_eol)
	{
	  now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	  nowtok = now->tok;
	  continue;
	}

      switch (nowtok)
	{
	case tok_charclass:
	  now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	  while (now->tok == tok_ident || now->tok == tok_string)
	    {
	      ctype_class_new (ldfile, ctype, now->val.str.startmb);
	      now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	      if (now->tok != tok_semicolon)
		break;
	      now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	    }
	  if (now->tok != tok_eol)
	    SYNTAX_ERROR (_("\
%s: syntax error in definition of new character class"), "LC_CTYPE");
	  break;

	case tok_charconv:
	  now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	  while (now->tok == tok_ident || now->tok == tok_string)
	    {
	      ctype_map_new (ldfile, ctype, now->val.str.startmb, charmap);
	      now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	      if (now->tok != tok_semicolon)
		break;
	      now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	    }
	  if (now->tok != tok_eol)
	    SYNTAX_ERROR (_("\
%s: syntax error in definition of new character map"), "LC_CTYPE");
	  break;

	case tok_class:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  /* We simply forget the `class' keyword and use the following
	     operand to determine the bit.  */
	  now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	  if (now->tok == tok_ident || now->tok == tok_string)
	    {
	      /* Must can be one of the predefined class names.  */
	      for (cnt = 0; cnt < ctype->nr_charclass; ++cnt)
		if (strcmp (ctype->classnames[cnt], now->val.str.startmb) == 0)
		  break;
	      if (cnt >= ctype->nr_charclass)
		{
		  /* OK, it's a new class.  */
		  ctype_class_new (ldfile, ctype, now->val.str.startmb);

		  class_bit = _ISwbit (ctype->nr_charclass - 1);
		}
	      else
		{
		  class_bit = _ISwbit (cnt);

		  free (now->val.str.startmb);
		}
	    }
	  else if (now->tok == tok_digit)
	    goto handle_tok_digit;
	  else if (now->tok < tok_upper || now->tok > tok_blank)
	    goto err_label;
	  else
	    {
	      class_bit = BITw (now->tok);
	      class256_bit = BIT (now->tok);
	    }

	  /* The next character must be a semicolon.  */
	  now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	  if (now->tok != tok_semicolon)
	    goto err_label;
	  goto read_charclass;

	case tok_upper:
	case tok_lower:
	case tok_alpha:
	case tok_alnum:
	case tok_space:
	case tok_cntrl:
	case tok_punct:
	case tok_graph:
	case tok_print:
	case tok_xdigit:
	case tok_blank:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  class_bit = BITw (now->tok);
	  class256_bit = BIT (now->tok);
	  handle_digits = 0;
	read_charclass:
	  ctype->class_done |= class_bit;
	  last_token = tok_none;
	  ellipsis_token = tok_none;
	  step = 1;
	  now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	  while (now->tok != tok_eol && now->tok != tok_eof)
	    {
	      uint32_t wch;
	      struct charseq *seq;

	      if (ellipsis_token == tok_none)
		{
		  if (get_character (now, charmap, repertoire, &seq, &wch))
		    goto err_label;

		  if (!ignore_content && seq != NULL && seq->nbytes == 1)
		    /* Yep, we can store information about this byte
		       sequence.  */
		    ctype->class256_collection[seq->bytes[0]] |= class256_bit;

		  if (!ignore_content && wch != ILLEGAL_CHAR_VALUE
		      && class_bit != 0)
		    /* We have the UCS4 position.  */
		    *find_idx (ctype, &ctype->class_collection,
			       &ctype->class_collection_max,
			       &ctype->class_collection_act, wch) |= class_bit;

		  last_token = now->tok;
		  /* Terminate the string.  */
		  if (last_token == tok_bsymbol)
		    {
		      now->val.str.startmb[now->val.str.lenmb] = '\0';
		      last_str = now->val.str.startmb;
		    }
		  else
		    last_str = NULL;
		  last_wch = wch;
		  memcpy (last_charcode, now->val.charcode.bytes, 16);
		  last_charcode_len = now->val.charcode.nbytes;

		  if (!ignore_content && handle_digits == 1)
		    {
		      /* We must store the digit values.  */
		      if (ctype->mbdigits_act == ctype->mbdigits_max)
			{
			  ctype->mbdigits_max += 10;
			  ctype->mbdigits = xrealloc (ctype->mbdigits,
						      (ctype->mbdigits_max
						       * sizeof (char *)));
			  ctype->wcdigits_max += 10;
			  ctype->wcdigits = xrealloc (ctype->wcdigits,
						      (ctype->wcdigits_max
						       * sizeof (uint32_t)));
			}

		      ctype->mbdigits[ctype->mbdigits_act++] = seq;
		      ctype->wcdigits[ctype->wcdigits_act++] = wch;
		    }
		  else if (!ignore_content && handle_digits == 2)
		    {
		      /* We must store the digit values.  */
		      if (ctype->outdigits_act >= 10)
			{
			  lr_error (ldfile, _("\
%s: field `%s' does not contain exactly ten entries"),
			    "LC_CTYPE", "outdigit");
			  lr_ignore_rest (ldfile, 0);
			  break;
			}

		      ctype->mboutdigits[ctype->outdigits_act] = seq;
		      ctype->wcoutdigits[ctype->outdigits_act] = wch;
		      ++ctype->outdigits_act;
		    }
		}
	      else
		{
		  /* Now it gets complicated.  We have to resolve the
		     ellipsis problem.  First we must distinguish between
		     the different kind of ellipsis and this must match the
		     tokens we have seen.  */
		  assert (last_token != tok_none);

		  if (last_token != now->tok)
		    {
		      lr_error (ldfile, _("\
ellipsis range must be marked by two operands of same type"));
		      lr_ignore_rest (ldfile, 0);
		      break;
		    }

		  if (last_token == tok_bsymbol)
		    {
		      if (ellipsis_token == tok_ellipsis3)
			lr_error (ldfile, _("with symbolic name range values \
the absolute ellipsis `...' must not be used"));

		      charclass_symbolic_ellipsis (ldfile, ctype, charmap,
						   repertoire, now, last_str,
						   class256_bit, class_bit,
						   (ellipsis_token
						    == tok_ellipsis4
						    ? 10 : 16),
						   ignore_content,
						   handle_digits, step);
		    }
		  else if (last_token == tok_ucs4)
		    {
		      if (ellipsis_token != tok_ellipsis2)
			lr_error (ldfile, _("\
with UCS range values one must use the hexadecimal symbolic ellipsis `..'"));

		      charclass_ucs4_ellipsis (ldfile, ctype, charmap,
					       repertoire, now, last_wch,
					       class256_bit, class_bit,
					       ignore_content, handle_digits,
					       step);
		    }
		  else
		    {
		      assert (last_token == tok_charcode);

		      if (ellipsis_token != tok_ellipsis3)
			lr_error (ldfile, _("\
with character code range values one must use the absolute ellipsis `...'"));

		      charclass_charcode_ellipsis (ldfile, ctype, charmap,
						   repertoire, now,
						   last_charcode,
						   last_charcode_len,
						   class256_bit, class_bit,
						   ignore_content,
						   handle_digits);
		    }

		  /* Now we have used the last value.  */
		  last_token = tok_none;
		}

	      /* Next we expect a semicolon or the end of the line.  */
	      now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	      if (now->tok == tok_eol || now->tok == tok_eof)
		break;

	      if (last_token != tok_none
		  && now->tok >= tok_ellipsis2 && now->tok <= tok_ellipsis4_2)
		{
		  if (now->tok == tok_ellipsis2_2)
		    {
		      now->tok = tok_ellipsis2;
		      step = 2;
		    }
		  else if (now->tok == tok_ellipsis4_2)
		    {
		      now->tok = tok_ellipsis4;
		      step = 2;
		    }

		  ellipsis_token = now->tok;

		  now = lr_token (ldfile, charmap, NULL, NULL, verbose);
		  continue;
		}

	      if (now->tok != tok_semicolon)
		goto err_label;

	      /* And get the next character.  */
	      now = lr_token (ldfile, charmap, NULL, NULL, verbose);

	      ellipsis_token = tok_none;
	      step = 1;
	    }
	  break;

	case tok_digit:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	handle_tok_digit:
	  class_bit = _ISwdigit;
	  class256_bit = _ISdigit;
	  handle_digits = 1;
	  goto read_charclass;

	case tok_outdigit:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  if (ctype->outdigits_act != 0)
	    lr_error (ldfile, _("\
%s: field `%s' declared more than once"),
		      "LC_CTYPE", "outdigit");
	  class_bit = 0;
	  class256_bit = 0;
	  handle_digits = 2;
	  goto read_charclass;

	case tok_toupper:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  mapidx = 0;
	  goto read_mapping;

	case tok_tolower:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  mapidx = 1;
	  goto read_mapping;

	case tok_map:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  /* We simply forget the `map' keyword and use the following
	     operand to determine the mapping.  */
	  now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	  if (now->tok == tok_ident || now->tok == tok_string)
	    {
	      size_t cnt;

	      for (cnt = 2; cnt < ctype->map_collection_nr; ++cnt)
		if (strcmp (now->val.str.startmb, ctype->mapnames[cnt]) == 0)
		  break;

	      if (cnt < ctype->map_collection_nr)
		free (now->val.str.startmb);
	      else
		/* OK, it's a new map.  */
		ctype_map_new (ldfile, ctype, now->val.str.startmb, charmap);

	      mapidx = cnt;
	    }
	  else if (now->tok < tok_toupper || now->tok > tok_tolower)
	    goto err_label;
	  else
	    mapidx = now->tok - tok_toupper;

	  now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	  /* This better should be a semicolon.  */
	  if (now->tok != tok_semicolon)
	    goto err_label;

	read_mapping:
	  /* Test whether this mapping was already defined.  */
	  if (ctype->tomap_done[mapidx])
	    {
	      lr_error (ldfile, _("duplicated definition for mapping `%s'"),
			ctype->mapnames[mapidx]);
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }
	  ctype->tomap_done[mapidx] = 1;

	  now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	  while (now->tok != tok_eol && now->tok != tok_eof)
	    {
	      struct charseq *from_seq;
	      uint32_t from_wch;
	      struct charseq *to_seq;
	      uint32_t to_wch;

	      /* Every pair starts with an opening brace.  */
	      if (now->tok != tok_open_brace)
		goto err_label;

	      /* Next comes the from-value.  */
	      now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	      if (get_character (now, charmap, repertoire, &from_seq,
				 &from_wch) != 0)
		goto err_label;

	      /* The next is a comma.  */
	      now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	      if (now->tok != tok_comma)
		goto err_label;

	      /* And the other value.  */
	      now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	      if (get_character (now, charmap, repertoire, &to_seq,
				 &to_wch) != 0)
		goto err_label;

	      /* And the last thing is the closing brace.  */
	      now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	      if (now->tok != tok_close_brace)
		goto err_label;

	      if (!ignore_content)
		{
		  /* Check whether the mapping converts from an ASCII value
		     to a non-ASCII value.  */
		  if (from_seq != NULL && from_seq->nbytes == 1
		      && isascii (from_seq->bytes[0])
		      && to_seq != NULL && (to_seq->nbytes != 1
					    || !isascii (to_seq->bytes[0])))
		    ctype->to_nonascii = 1;

		  if (mapidx < 2 && from_seq != NULL && to_seq != NULL
		      && from_seq->nbytes == 1 && to_seq->nbytes == 1)
		    /* We can use this value.  */
		    ctype->map256_collection[mapidx][from_seq->bytes[0]]
		      = to_seq->bytes[0];

		  if (from_wch != ILLEGAL_CHAR_VALUE
		      && to_wch != ILLEGAL_CHAR_VALUE)
		    /* Both correct values.  */
		    *find_idx (ctype, &ctype->map_collection[mapidx],
			       &ctype->map_collection_max[mapidx],
			       &ctype->map_collection_act[mapidx],
			       from_wch) = to_wch;
		}

	      /* Now comes a semicolon or the end of the line/file.  */
	      now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	      if (now->tok == tok_semicolon)
		now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	    }
	  break;

	case tok_translit_start:
	  /* Ignore the entire translit section with its peculiar syntax
	     if we don't need the input.  */
	  if (ignore_content)
	    {
	      do
		{
		  lr_ignore_rest (ldfile, 0);
		  now = lr_token (ldfile, charmap, NULL, NULL, verbose);
		}
	      while (now->tok != tok_translit_end && now->tok != tok_eof);

	      if (now->tok == tok_eof)
		lr_error (ldfile, _(\
"%s: `translit_start' section does not end with `translit_end'"),
			  "LC_CTYPE");

	      break;
	    }

	  /* The rest of the line better should be empty.  */
	  lr_ignore_rest (ldfile, 1);

	  /* We count here the number of allocated entries in the `translit'
	     array.  */
	  cnt = 0;

	  ldfile->translate_strings = 1;
	  ldfile->return_widestr = 1;

	  /* We proceed until we see the `translit_end' token.  */
	  while (now = lr_token (ldfile, charmap, NULL, repertoire, verbose),
		 now->tok != tok_translit_end && now->tok != tok_eof)
	    {
	      if (now->tok == tok_eol)
		/* Ignore empty lines.  */
		continue;

	      if (now->tok == tok_include)
		{
		  /* We have to include locale.  */
		  const char *locale_name;
		  const char *repertoire_name;
		  struct translit_include_t *include_stmt, **include_ptr;

		  now = lr_token (ldfile, charmap, NULL, NULL, verbose);
		  /* This should be a string or an identifier.  In any
		     case something to name a locale.  */
		  if (now->tok != tok_string && now->tok != tok_ident)
		    {
		    translit_syntax:
		      lr_error (ldfile, _("%s: syntax error"), "LC_CTYPE");
		      lr_ignore_rest (ldfile, 0);
		      continue;
		    }
		  locale_name = now->val.str.startmb;

		  /* Next should be a semicolon.  */
		  now = lr_token (ldfile, charmap, NULL, NULL, verbose);
		  if (now->tok != tok_semicolon)
		    goto translit_syntax;

		  /* Now the repertoire name.  */
		  now = lr_token (ldfile, charmap, NULL, NULL, verbose);
		  if ((now->tok != tok_string && now->tok != tok_ident)
		      || now->val.str.startmb == NULL)
		    goto translit_syntax;
		  repertoire_name = now->val.str.startmb;
		  if (repertoire_name[0] == '\0')
		    /* Ignore the empty string.  */
		    repertoire_name = NULL;

		  /* Save the include statement for later processing.  */
		  include_stmt = (struct translit_include_t *)
		    xmalloc (sizeof (struct translit_include_t));
		  include_stmt->copy_locale = locale_name;
		  include_stmt->copy_repertoire = repertoire_name;
		  include_stmt->next = NULL;

		  include_ptr = &ctype->translit_include;
		  while (*include_ptr != NULL)
		    include_ptr = &(*include_ptr)->next;
		  *include_ptr = include_stmt;

		  /* The rest of the line must be empty.  */
		  lr_ignore_rest (ldfile, 1);

		  /* Make sure the locale is read.  */
		  add_to_readlist (LC_CTYPE, locale_name, repertoire_name,
				   1, NULL);
		  continue;
		}
	      else if (now->tok == tok_default_missing)
		{
		  uint32_t *wstr;

		  while (1)
		    {
		      /* We expect a single character or string as the
			 argument.  */
		      now = lr_token (ldfile, charmap, NULL, NULL, verbose);
		      wstr = read_widestring (ldfile, now, charmap,
					      repertoire);

		      if (wstr != NULL)
			{
			  if (ctype->default_missing != NULL)
			    {
			      lr_error (ldfile, _("\
%s: duplicate `default_missing' definition"), "LC_CTYPE");
			      record_error_at_line (0, 0,
						    ctype->default_missing_file,
						    ctype->default_missing_lineno,
						    _("\
previous definition was here"));
			    }
			  else
			    {
			      ctype->default_missing = wstr;
			      ctype->default_missing_file = ldfile->fname;
			      ctype->default_missing_lineno = ldfile->lineno;
			    }
			  /* We can have more entries, ignore them.  */
			  lr_ignore_rest (ldfile, 0);
			  break;
			}
		      else if (wstr == (uint32_t *) -1l)
			/* This was an syntax error.  */
			break;

		      /* Maybe there is another replacement we can use.  */
		      now = lr_token (ldfile, charmap, NULL, NULL, verbose);
		      if (now->tok == tok_eol || now->tok == tok_eof)
			{
			  /* Nothing found.  We tell the user.  */
			  lr_error (ldfile, _("\
%s: no representable `default_missing' definition found"), "LC_CTYPE");
			  break;
			}
		      if (now->tok != tok_semicolon)
			goto translit_syntax;
		    }

		  continue;
		}
	      else if (now->tok == tok_translit_ignore)
		{
		  read_translit_ignore_entry (ldfile, ctype, charmap,
					      repertoire);
		  continue;
		}

	      read_translit_entry (ldfile, ctype, now, charmap, repertoire);
	    }
	  ldfile->return_widestr = 0;

	  if (now->tok == tok_eof)
	    lr_error (ldfile, _(\
"%s: `translit_start' section does not end with `translit_end'"),
		      "LC_CTYPE");

	  break;

	case tok_ident:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  /* This could mean one of several things.  First test whether
	     it's a character class name.  */
	  for (cnt = 0; cnt < ctype->nr_charclass; ++cnt)
	    if (strcmp (now->val.str.startmb, ctype->classnames[cnt]) == 0)
	      break;
	  if (cnt < ctype->nr_charclass)
	    {
	      class_bit = _ISwbit (cnt);
	      class256_bit = cnt <= 11 ? _ISbit (cnt) : 0;
	      free (now->val.str.startmb);
	      goto read_charclass;
	    }
	  for (cnt = 0; cnt < ctype->map_collection_nr; ++cnt)
	    if (strcmp (now->val.str.startmb, ctype->mapnames[cnt]) == 0)
	      break;
	  if (cnt < ctype->map_collection_nr)
	    {
	      mapidx = cnt;
	      free (now->val.str.startmb);
	      goto read_mapping;
            }
	  break;

	case tok_end:
	  /* Next we assume `LC_CTYPE'.  */
	  now = lr_token (ldfile, charmap, NULL, NULL, verbose);
	  if (now->tok == tok_eof)
	    break;
	  if (now->tok == tok_eol)
	    lr_error (ldfile, _("%s: incomplete `END' line"),
		      "LC_CTYPE");
	  else if (now->tok != tok_lc_ctype)
	    lr_error (ldfile, _("\
%1$s: definition does not end with `END %1$s'"), "LC_CTYPE");
	  lr_ignore_rest (ldfile, now->tok == tok_lc_ctype);
	  return;

	default:
	err_label:
	  if (now->tok != tok_eof)
	    SYNTAX_ERROR (_("%s: syntax error"), "LC_CTYPE");
	}

      /* Prepare for the next round.  */
      now = lr_token (ldfile, charmap, NULL, NULL, verbose);
      nowtok = now->tok;
    }

  /* When we come here we reached the end of the file.  */
  lr_error (ldfile, _("%s: premature end of file"), "LC_CTYPE");
}


/* Subroutine of set_class_defaults, below.  */
static void
set_one_default (struct locale_ctype_t *ctype,
                 const struct charmap_t *charmap,
                 int bitpos, int from, int to)
{
  char tmp[2];
  int ch;
  int bit = _ISbit (bitpos);
  int bitw = _ISwbit (bitpos);
  /* Define string.  */
  strcpy (tmp, "?");

  for (ch = from; ch <= to; ++ch)
    {
      struct charseq *seq;
      tmp[0] = ch;

      seq = charmap_find_value (charmap, tmp, 1);
      if (seq == NULL)
        {
          char buf[10];
          sprintf (buf, "U%08X", ch);
          seq = charmap_find_value (charmap, buf, 9);
        }
      if (seq == NULL)
        {
          record_error (0, 0, _("\
%s: character `%s' not defined while needed as default value"),
			"LC_CTYPE", tmp);
        }
      else if (seq->nbytes != 1)
	record_error (0, 0, _("\
%s: character `%s' in charmap not representable with one byte"),
		      "LC_CTYPE", tmp);
      else
        ctype->class256_collection[seq->bytes[0]] |= bit;

      /* No need to search here, the ASCII value is also the Unicode
         value.  */
      ELEM (ctype, class_collection, , ch) |= bitw;
    }
}

static void
set_class_defaults (struct locale_ctype_t *ctype,
		    const struct charmap_t *charmap,
		    struct repertoire_t *repertoire)
{
#define set_default(bitpos, from, to) \
  set_one_default (ctype, charmap, bitpos, from, to)

  /* These function defines the default values for the classes and conversions
     according to POSIX.2 2.5.2.1.
     It may seem that the order of these if-blocks is arbitrary but it is NOT.
     Don't move them unless you know what you do!  */

  /* Set default values if keyword was not present.  */
  if ((ctype->class_done & BITw (tok_upper)) == 0)
    /* "If this keyword [lower] is not specified, the lowercase letters
        `A' through `Z', ..., shall automatically belong to this class,
	with implementation defined character values."  [P1003.2, 2.5.2.1]  */
    set_default (BITPOS (tok_upper), 'A', 'Z');

  if ((ctype->class_done & BITw (tok_lower)) == 0)
    /* "If this keyword [lower] is not specified, the lowercase letters
        `a' through `z', ..., shall automatically belong to this class,
	with implementation defined character values."  [P1003.2, 2.5.2.1]  */
    set_default (BITPOS (tok_lower), 'a', 'z');

  if ((ctype->class_done & BITw (tok_alpha)) == 0)
    {
      /* Table 2-6 in P1003.2 says that characters in class `upper' or
	 class `lower' *must* be in class `alpha'.  */
      unsigned long int mask = BIT (tok_upper) | BIT (tok_lower);
      unsigned long int maskw = BITw (tok_upper) | BITw (tok_lower);

      for (size_t cnt = 0; cnt < 256; ++cnt)
	if ((ctype->class256_collection[cnt] & mask) != 0)
	  ctype->class256_collection[cnt] |= BIT (tok_alpha);

      for (size_t cnt = 0; cnt < ctype->class_collection_act; ++cnt)
	if ((ctype->class_collection[cnt] & maskw) != 0)
	  ctype->class_collection[cnt] |= BITw (tok_alpha);
    }

  if ((ctype->class_done & BITw (tok_digit)) == 0)
    /* "If this keyword [digit] is not specified, the digits `0' through
        `9', ..., shall automatically belong to this class, with
	implementation-defined character values."  [P1003.2, 2.5.2.1]  */
    set_default (BITPOS (tok_digit), '0', '9');

  /* "Only characters specified for the `alpha' and `digit' keyword
     shall be specified.  Characters specified for the keyword `alpha'
     and `digit' are automatically included in this class.  */
  {
    unsigned long int mask = BIT (tok_alpha) | BIT (tok_digit);
    unsigned long int maskw = BITw (tok_alpha) | BITw (tok_digit);

    for (size_t cnt = 0; cnt < 256; ++cnt)
      if ((ctype->class256_collection[cnt] & mask) != 0)
	ctype->class256_collection[cnt] |= BIT (tok_alnum);

    for (size_t cnt = 0; cnt < ctype->class_collection_act; ++cnt)
      if ((ctype->class_collection[cnt] & maskw) != 0)
	ctype->class_collection[cnt] |= BITw (tok_alnum);
  }

  if ((ctype->class_done & BITw (tok_space)) == 0)
    /* "If this keyword [space] is not specified, the characters <space>,
        <form-feed>, <newline>, <carriage-return>, <tab>, and
	<vertical-tab>, ..., shall automatically belong to this class,
	with implementation-defined character values."  [P1003.2, 2.5.2.1]  */
    {
      struct charseq *seq;

      seq = charmap_find_value (charmap, "space", 5);
      if (seq == NULL)
	seq = charmap_find_value (charmap, "SP", 2);
      if (seq == NULL)
	seq = charmap_find_value (charmap, "U00000020", 9);
      if (seq == NULL)
	{
	  record_error (0, 0, _("\
%s: character `%s' not defined while needed as default value"),
			"LC_CTYPE", "<space>");
	}
      else if (seq->nbytes != 1)
	record_error (0, 0, _("\
%s: character `%s' in charmap not representable with one byte"),
		      "LC_CTYPE", "<space>");
      else
	ctype->class256_collection[seq->bytes[0]] |= BIT (tok_space);

      /* No need to search.  */
      ELEM (ctype, class_collection, , L' ') |= BITw (tok_space);

      seq = charmap_find_value (charmap, "form-feed", 9);
      if (seq == NULL)
	seq = charmap_find_value (charmap, "U0000000C", 9);
      if (seq == NULL)
	{
	  record_error (0, 0, _("\
%s: character `%s' not defined while needed as default value"),
				    "LC_CTYPE", "<form-feed>");
	}
      else if (seq->nbytes != 1)
	record_error (0, 0, _("\
%s: character `%s' in charmap not representable with one byte"),
		      "LC_CTYPE", "<form-feed>");
      else
	ctype->class256_collection[seq->bytes[0]] |= BIT (tok_space);

      /* No need to search.  */
      ELEM (ctype, class_collection, , L'\f') |= BITw (tok_space);


      seq = charmap_find_value (charmap, "newline", 7);
      if (seq == NULL)
	seq = charmap_find_value (charmap, "U0000000A", 9);
      if (seq == NULL)
	{
	  record_error (0, 0, _("\
%s: character `%s' not defined while needed as default value"),
			"LC_CTYPE", "<newline>");
	}
      else if (seq->nbytes != 1)
	record_error (0, 0, _("\
%s: character `%s' in charmap not representable with one byte"),
		      "LC_CTYPE", "<newline>");
      else
	ctype->class256_collection[seq->bytes[0]] |= BIT (tok_space);

      /* No need to search.  */
      ELEM (ctype, class_collection, , L'\n') |= BITw (tok_space);


      seq = charmap_find_value (charmap, "carriage-return", 15);
      if (seq == NULL)
	seq = charmap_find_value (charmap, "U0000000D", 9);
      if (seq == NULL)
	{
	  record_error (0, 0, _("\
%s: character `%s' not defined while needed as default value"),
			"LC_CTYPE", "<carriage-return>");
	}
      else if (seq->nbytes != 1)
	record_error (0, 0, _("\
%s: character `%s' in charmap not representable with one byte"),
		      "LC_CTYPE", "<carriage-return>");
      else
	ctype->class256_collection[seq->bytes[0]] |= BIT (tok_space);

      /* No need to search.  */
      ELEM (ctype, class_collection, , L'\r') |= BITw (tok_space);


      seq = charmap_find_value (charmap, "tab", 3);
      if (seq == NULL)
	seq = charmap_find_value (charmap, "U00000009", 9);
      if (seq == NULL)
	{
	  record_error (0, 0, _("\
%s: character `%s' not defined while needed as default value"),
			"LC_CTYPE", "<tab>");
	}
      else if (seq->nbytes != 1)
	record_error (0, 0, _("\
%s: character `%s' in charmap not representable with one byte"),
		      "LC_CTYPE", "<tab>");
      else
	ctype->class256_collection[seq->bytes[0]] |= BIT (tok_space);

      /* No need to search.  */
      ELEM (ctype, class_collection, , L'\t') |= BITw (tok_space);


      seq = charmap_find_value (charmap, "vertical-tab", 12);
      if (seq == NULL)
	seq = charmap_find_value (charmap, "U0000000B", 9);
      if (seq == NULL)
	{
	  record_error (0, 0, _("\
%s: character `%s' not defined while needed as default value"),
			"LC_CTYPE", "<vertical-tab>");
	}
      else if (seq->nbytes != 1)
	record_error (0, 0, _("\
%s: character `%s' in charmap not representable with one byte"),
		      "LC_CTYPE", "<vertical-tab>");
      else
	ctype->class256_collection[seq->bytes[0]] |= BIT (tok_space);

      /* No need to search.  */
      ELEM (ctype, class_collection, , L'\v') |= BITw (tok_space);
    }

  if ((ctype->class_done & BITw (tok_xdigit)) == 0)
    /* "If this keyword is not specified, the digits `0' to `9', the
        uppercase letters `A' through `F', and the lowercase letters `a'
	through `f', ..., shell automatically belong to this class, with
	implementation defined character values."  [P1003.2, 2.5.2.1]  */
    {
      set_default (BITPOS (tok_xdigit), '0', '9');
      set_default (BITPOS (tok_xdigit), 'A', 'F');
      set_default (BITPOS (tok_xdigit), 'a', 'f');
    }

  if ((ctype->class_done & BITw (tok_blank)) == 0)
    /* "If this keyword [blank] is unspecified, the characters <space> and
       <tab> shall belong to this character class."  [P1003.2, 2.5.2.1]  */
   {
      struct charseq *seq;

      seq = charmap_find_value (charmap, "space", 5);
      if (seq == NULL)
	seq = charmap_find_value (charmap, "SP", 2);
      if (seq == NULL)
	seq = charmap_find_value (charmap, "U00000020", 9);
      if (seq == NULL)
	{
	  record_error (0, 0, _("\
%s: character `%s' not defined while needed as default value"),
			"LC_CTYPE", "<space>");
	}
      else if (seq->nbytes != 1)
	record_error (0, 0, _("\
%s: character `%s' in charmap not representable with one byte"),
		      "LC_CTYPE", "<space>");
      else
	ctype->class256_collection[seq->bytes[0]] |= BIT (tok_blank);

      /* No need to search.  */
      ELEM (ctype, class_collection, , L' ') |= BITw (tok_blank);


      seq = charmap_find_value (charmap, "tab", 3);
      if (seq == NULL)
	seq = charmap_find_value (charmap, "U00000009", 9);
      if (seq == NULL)
	{
	   record_error (0, 0, _("\
%s: character `%s' not defined while needed as default value"),
		         "LC_CTYPE", "<tab>");
	}
      else if (seq->nbytes != 1)
	record_error (0, 0, _("\
%s: character `%s' in charmap not representable with one byte"),
		      "LC_CTYPE", "<tab>");
      else
	ctype->class256_collection[seq->bytes[0]] |= BIT (tok_blank);

      /* No need to search.  */
      ELEM (ctype, class_collection, , L'\t') |= BITw (tok_blank);
    }

  if ((ctype->class_done & BITw (tok_graph)) == 0)
    /* "If this keyword [graph] is not specified, characters specified for
        the keywords `upper', `lower', `alpha', `digit', `xdigit' and `punct',
	shall belong to this character class."  [P1003.2, 2.5.2.1]  */
    {
      unsigned long int mask = BIT (tok_upper) | BIT (tok_lower)
	| BIT (tok_alpha) | BIT (tok_digit) | BIT (tok_xdigit)
	| BIT (tok_punct);
      unsigned long int maskw = BITw (tok_upper) | BITw (tok_lower)
	| BITw (tok_alpha) | BITw (tok_digit) | BITw (tok_xdigit)
	| BITw (tok_punct);

      for (size_t cnt = 0; cnt < ctype->class_collection_act; ++cnt)
	if ((ctype->class_collection[cnt] & maskw) != 0)
	  ctype->class_collection[cnt] |= BITw (tok_graph);

      for (size_t cnt = 0; cnt < 256; ++cnt)
	if ((ctype->class256_collection[cnt] & mask) != 0)
	  ctype->class256_collection[cnt] |= BIT (tok_graph);
    }

  if ((ctype->class_done & BITw (tok_print)) == 0)
    /* "If this keyword [print] is not provided, characters specified for
        the keywords `upper', `lower', `alpha', `digit', `xdigit', `punct',
	and the <space> character shall belong to this character class."
	[P1003.2, 2.5.2.1]  */
    {
      unsigned long int mask = BIT (tok_upper) | BIT (tok_lower)
	| BIT (tok_alpha) | BIT (tok_digit) | BIT (tok_xdigit)
	| BIT (tok_punct);
      unsigned long int maskw = BITw (tok_upper) | BITw (tok_lower)
	| BITw (tok_alpha) | BITw (tok_digit) | BITw (tok_xdigit)
	| BITw (tok_punct);
      struct charseq *seq;

      for (size_t cnt = 0; cnt < ctype->class_collection_act; ++cnt)
	if ((ctype->class_collection[cnt] & maskw) != 0)
	  ctype->class_collection[cnt] |= BITw (tok_print);

      for (size_t cnt = 0; cnt < 256; ++cnt)
	if ((ctype->class256_collection[cnt] & mask) != 0)
	  ctype->class256_collection[cnt] |= BIT (tok_print);


      seq = charmap_find_value (charmap, "space", 5);
      if (seq == NULL)
	seq = charmap_find_value (charmap, "SP", 2);
      if (seq == NULL)
	seq = charmap_find_value (charmap, "U00000020", 9);
      if (seq == NULL)
	{
	  record_error (0, 0, _("\
%s: character `%s' not defined while needed as default value"),
			"LC_CTYPE", "<space>");
	}
      else if (seq->nbytes != 1)
	record_error (0, 0, _("\
%s: character `%s' in charmap not representable with one byte"),
		      "LC_CTYPE", "<space>");
      else
	ctype->class256_collection[seq->bytes[0]] |= BIT (tok_print);

      /* No need to search.  */
      ELEM (ctype, class_collection, , L' ') |= BITw (tok_print);
    }

  if (ctype->tomap_done[0] == 0)
    /* "If this keyword [toupper] is not specified, the lowercase letters
        `a' through `z', and their corresponding uppercase letters `A' to
	`Z', ..., shall automatically be included, with implementation-
	defined character values."  [P1003.2, 2.5.2.1]  */
    {
      char tmp[4];
      int ch;

      strcpy (tmp, "<?>");

      for (ch = 'a'; ch <= 'z'; ++ch)
	{
	  struct charseq *seq_from, *seq_to;

	  tmp[1] = (char) ch;

	  seq_from = charmap_find_value (charmap, &tmp[1], 1);
	  if (seq_from == NULL)
	    {
	      char buf[10];
	      sprintf (buf, "U%08X", ch);
	      seq_from = charmap_find_value (charmap, buf, 9);
	    }
	  if (seq_from == NULL)
	    {
	      record_error (0, 0, _("\
%s: character `%s' not defined while needed as default value"),
			    "LC_CTYPE", tmp);
	    }
	  else if (seq_from->nbytes != 1)
	    {
	      record_error (0, 0, _("\
%s: character `%s' needed as default value not representable with one byte"),
			    "LC_CTYPE", tmp);
	    }
	  else
	    {
	      /* This conversion is implementation defined.  */
	      tmp[1] = (char) (ch + ('A' - 'a'));
	      seq_to = charmap_find_value (charmap, &tmp[1], 1);
	      if (seq_to == NULL)
		{
		  char buf[10];
		  sprintf (buf, "U%08X", ch + ('A' - 'a'));
		  seq_to = charmap_find_value (charmap, buf, 9);
		}
	      if (seq_to == NULL)
		{
		  record_error (0, 0, _("\
%s: character `%s' not defined while needed as default value"),
				"LC_CTYPE", tmp);
		}
	      else if (seq_to->nbytes != 1)
		{
		  record_error (0, 0, _("\
%s: character `%s' needed as default value not representable with one byte"),
				"LC_CTYPE", tmp);
		}
	      else
		/* The index [0] is determined by the order of the
		   `ctype_map_newP' calls in `ctype_startup'.  */
		ctype->map256_collection[0][seq_from->bytes[0]]
		  = seq_to->bytes[0];
	    }

	  /* No need to search.  */
	  ELEM (ctype, map_collection, [0], ch) = ch + ('A' - 'a');
	}
    }

  if (ctype->tomap_done[1] == 0)
    /* "If this keyword [tolower] is not specified, the mapping shall be
       the reverse mapping of the one specified to `toupper'."  [P1003.2]  */
    {
      for (size_t cnt = 0; cnt < ctype->map_collection_act[0]; ++cnt)
	if (ctype->map_collection[0][cnt] != 0)
	  ELEM (ctype, map_collection, [1],
		ctype->map_collection[0][cnt])
	    = ctype->charnames[cnt];

      for (size_t cnt = 0; cnt < 256; ++cnt)
	if (ctype->map256_collection[0][cnt] != 0)
	  ctype->map256_collection[1][ctype->map256_collection[0][cnt]] = cnt;
    }

  if (ctype->outdigits_act != 10)
    {
      if (ctype->outdigits_act != 0)
	record_error (0, 0, _("\
%s: field `%s' does not contain exactly ten entries"),
		      "LC_CTYPE", "outdigit");

      for (size_t cnt = ctype->outdigits_act; cnt < 10; ++cnt)
	{
	  ctype->mboutdigits[cnt] = charmap_find_symbol (charmap,
							 (char *) digits + cnt,
							 1);

	  if (ctype->mboutdigits[cnt] == NULL)
	    ctype->mboutdigits[cnt] = charmap_find_symbol (charmap,
							   longnames[cnt],
							   strlen (longnames[cnt]));

	  if (ctype->mboutdigits[cnt] == NULL)
	    ctype->mboutdigits[cnt] = charmap_find_symbol (charmap,
							   uninames[cnt], 9);

	  if (ctype->mboutdigits[cnt] == NULL)
	    {
	      /* Provide a replacement.  */
	      record_error (0, 0, _("\
no output digits defined and none of the standard names in the charmap"));

	      ctype->mboutdigits[cnt] = obstack_alloc (&((struct charmap_t *) charmap)->mem_pool,
						       sizeof (struct charseq)
						       + 1);

	      /* This is better than nothing.  */
	      ctype->mboutdigits[cnt]->bytes[0] = digits[cnt];
	      ctype->mboutdigits[cnt]->nbytes = 1;
	    }

	  ctype->wcoutdigits[cnt] = L'0' + cnt;
	}

      ctype->outdigits_act = 10;
    }

#undef set_default
}


/* Initialize.  Assumes t->p and t->q have already been set.  */
static inline void
wctype_table_init (struct wctype_table *t)
{
  t->level1 = NULL;
  t->level1_alloc = t->level1_size = 0;
  t->level2 = NULL;
  t->level2_alloc = t->level2_size = 0;
  t->level3 = NULL;
  t->level3_alloc = t->level3_size = 0;
}

/* Retrieve an entry.  */
static inline int
wctype_table_get (struct wctype_table *t, uint32_t wc)
{
  uint32_t index1 = wc >> (t->q + t->p + 5);
  if (index1 < t->level1_size)
    {
      uint32_t lookup1 = t->level1[index1];
      if (lookup1 != EMPTY)
	{
	  uint32_t index2 = ((wc >> (t->p + 5)) & ((1 << t->q) - 1))
			    + (lookup1 << t->q);
	  uint32_t lookup2 = t->level2[index2];
	  if (lookup2 != EMPTY)
	    {
	      uint32_t index3 = ((wc >> 5) & ((1 << t->p) - 1))
				+ (lookup2 << t->p);
	      uint32_t lookup3 = t->level3[index3];
	      uint32_t index4 = wc & 0x1f;

	      return (lookup3 >> index4) & 1;
	    }
	}
    }
  return 0;
}

/* Add one entry.  */
static void
wctype_table_add (struct wctype_table *t, uint32_t wc)
{
  uint32_t index1 = wc >> (t->q + t->p + 5);
  uint32_t index2 = (wc >> (t->p + 5)) & ((1 << t->q) - 1);
  uint32_t index3 = (wc >> 5) & ((1 << t->p) - 1);
  uint32_t index4 = wc & 0x1f;
  size_t i, i1, i2;

  if (index1 >= t->level1_size)
    {
      if (index1 >= t->level1_alloc)
	{
	  size_t alloc = 2 * t->level1_alloc;
	  if (alloc <= index1)
	    alloc = index1 + 1;
	  t->level1 = (uint32_t *) xrealloc ((char *) t->level1,
					     alloc * sizeof (uint32_t));
	  t->level1_alloc = alloc;
	}
      while (index1 >= t->level1_size)
	t->level1[t->level1_size++] = EMPTY;
    }

  if (t->level1[index1] == EMPTY)
    {
      if (t->level2_size == t->level2_alloc)
	{
	  size_t alloc = 2 * t->level2_alloc + 1;
	  t->level2 = (uint32_t *) xrealloc ((char *) t->level2,
					     (alloc << t->q) * sizeof (uint32_t));
	  t->level2_alloc = alloc;
	}
      i1 = t->level2_size << t->q;
      i2 = (t->level2_size + 1) << t->q;
      for (i = i1; i < i2; i++)
	t->level2[i] = EMPTY;
      t->level1[index1] = t->level2_size++;
    }

  index2 += t->level1[index1] << t->q;

  if (t->level2[index2] == EMPTY)
    {
      if (t->level3_size == t->level3_alloc)
	{
	  size_t alloc = 2 * t->level3_alloc + 1;
	  t->level3 = (uint32_t *) xrealloc ((char *) t->level3,
					     (alloc << t->p) * sizeof (uint32_t));
	  t->level3_alloc = alloc;
	}
      i1 = t->level3_size << t->p;
      i2 = (t->level3_size + 1) << t->p;
      for (i = i1; i < i2; i++)
	t->level3[i] = 0;
      t->level2[index2] = t->level3_size++;
    }

  index3 += t->level2[index2] << t->p;

  t->level3[index3] |= (uint32_t)1 << index4;
}

/* Finalize and shrink.  */
static void
add_locale_wctype_table (struct locale_file *file, struct wctype_table *t)
{
  size_t i, j, k;
  uint32_t reorder3[t->level3_size];
  uint32_t reorder2[t->level2_size];
  uint32_t level2_offset, level3_offset;

  /* Uniquify level3 blocks.  */
  k = 0;
  for (j = 0; j < t->level3_size; j++)
    {
      for (i = 0; i < k; i++)
	if (memcmp (&t->level3[i << t->p], &t->level3[j << t->p],
		    (1 << t->p) * sizeof (uint32_t)) == 0)
	  break;
      /* Relocate block j to block i.  */
      reorder3[j] = i;
      if (i == k)
	{
	  if (i != j)
	    memcpy (&t->level3[i << t->p], &t->level3[j << t->p],
		    (1 << t->p) * sizeof (uint32_t));
	  k++;
	}
    }
  t->level3_size = k;

  for (i = 0; i < (t->level2_size << t->q); i++)
    if (t->level2[i] != EMPTY)
      t->level2[i] = reorder3[t->level2[i]];

  /* Uniquify level2 blocks.  */
  k = 0;
  for (j = 0; j < t->level2_size; j++)
    {
      for (i = 0; i < k; i++)
	if (memcmp (&t->level2[i << t->q], &t->level2[j << t->q],
		    (1 << t->q) * sizeof (uint32_t)) == 0)
	  break;
      /* Relocate block j to block i.  */
      reorder2[j] = i;
      if (i == k)
	{
	  if (i != j)
	    memcpy (&t->level2[i << t->q], &t->level2[j << t->q],
		    (1 << t->q) * sizeof (uint32_t));
	  k++;
	}
    }
  t->level2_size = k;

  for (i = 0; i < t->level1_size; i++)
    if (t->level1[i] != EMPTY)
      t->level1[i] = reorder2[t->level1[i]];

  t->result_size =
    5 * sizeof (uint32_t)
    + t->level1_size * sizeof (uint32_t)
    + (t->level2_size << t->q) * sizeof (uint32_t)
    + (t->level3_size << t->p) * sizeof (uint32_t);

  level2_offset =
    5 * sizeof (uint32_t)
    + t->level1_size * sizeof (uint32_t);
  level3_offset =
    5 * sizeof (uint32_t)
    + t->level1_size * sizeof (uint32_t)
    + (t->level2_size << t->q) * sizeof (uint32_t);

  start_locale_structure (file);
  add_locale_uint32 (file, t->q + t->p + 5);
  add_locale_uint32 (file, t->level1_size);
  add_locale_uint32 (file, t->p + 5);
  add_locale_uint32 (file, (1 << t->q) - 1);
  add_locale_uint32 (file, (1 << t->p) - 1);

  for (i = 0; i < t->level1_size; i++)
    add_locale_uint32
      (file,
       t->level1[i] == EMPTY
       ? 0
       : (t->level1[i] << t->q) * sizeof (uint32_t) + level2_offset);

  for (i = 0; i < (t->level2_size << t->q); i++)
    add_locale_uint32
      (file,
       t->level2[i] == EMPTY
       ? 0
       : (t->level2[i] << t->p) * sizeof (uint32_t) + level3_offset);

  add_locale_uint32_array (file, t->level3, t->level3_size << t->p);
  end_locale_structure (file);

  if (t->level1_alloc > 0)
    free (t->level1);
  if (t->level2_alloc > 0)
    free (t->level2);
  if (t->level3_alloc > 0)
    free (t->level3);
}

/* Flattens the included transliterations into a translit list.
   Inserts them in the list at `cursor', and returns the new cursor.  */
static struct translit_t **
translit_flatten (struct locale_ctype_t *ctype,
		  const struct charmap_t *charmap,
		  struct translit_t **cursor)
{
  while (ctype->translit_include != NULL)
    {
      const char *copy_locale = ctype->translit_include->copy_locale;
      const char *copy_repertoire = ctype->translit_include->copy_repertoire;
      struct localedef_t *other;

      /* Unchain the include statement.  During the depth-first traversal
	 we don't want to visit any locale more than once.  */
      ctype->translit_include = ctype->translit_include->next;

      other = find_locale (LC_CTYPE, copy_locale, copy_repertoire, charmap);

      if (other == NULL || other->categories[LC_CTYPE].ctype == NULL)
	{
	  record_error (0, 0, _("\
%s: transliteration data from locale `%s' not available"),
			"LC_CTYPE", copy_locale);
	}
      else
	{
	  struct locale_ctype_t *other_ctype =
	    other->categories[LC_CTYPE].ctype;

	  cursor = translit_flatten (other_ctype, charmap, cursor);
	  assert (other_ctype->translit_include == NULL);

	  if (other_ctype->translit != NULL)
	    {
	      /* Insert the other_ctype->translit list at *cursor.  */
	      struct translit_t *endp = other_ctype->translit;
	      while (endp->next != NULL)
		endp = endp->next;

	      endp->next = *cursor;
	      *cursor = other_ctype->translit;

	      /* Avoid any risk of circular lists.  */
	      other_ctype->translit = NULL;

	      cursor = &endp->next;
	    }

	  if (ctype->default_missing == NULL)
	    ctype->default_missing = other_ctype->default_missing;
	}
    }

  return cursor;
}

static void
allocate_arrays (struct locale_ctype_t *ctype, const struct charmap_t *charmap,
		 struct repertoire_t *repertoire)
{
  size_t idx, nr;
  const void *key;
  size_t len;
  void *vdata;
  void *curs;

  /* You wonder about this amount of memory?  This is only because some
     users do not manage to address the array with unsigned values or
     data types with range >= 256.  '\200' would result in the array
     index -128.  To help these poor people we duplicate the entries for
     128 up to 255 below the entry for \0.  */
  ctype->ctype_b = (char_class_t *) xcalloc (256 + 128, sizeof (char_class_t));
  ctype->ctype32_b = (char_class32_t *) xcalloc (256, sizeof (char_class32_t));
  ctype->class_b = (uint32_t **)
    xmalloc (ctype->nr_charclass * sizeof (uint32_t *));
  ctype->class_3level = (struct wctype_table *)
    xmalloc (ctype->nr_charclass * sizeof (struct wctype_table));

  /* This is the array accessed using the multibyte string elements.  */
  for (idx = 0; idx < 256; ++idx)
    ctype->ctype_b[128 + idx] = ctype->class256_collection[idx];

  /* Mirror first 127 entries.  We must take care that entry -1 is not
     mirrored because EOF == -1.  */
  for (idx = 0; idx < 127; ++idx)
    ctype->ctype_b[idx] = ctype->ctype_b[256 + idx];

  /* The 32 bit array contains all characters < 0x100.  */
  for (idx = 0; idx < ctype->class_collection_act; ++idx)
    if (ctype->charnames[idx] < 0x100)
      ctype->ctype32_b[ctype->charnames[idx]] = ctype->class_collection[idx];

  for (nr = 0; nr < ctype->nr_charclass; nr++)
    {
      ctype->class_b[nr] = (uint32_t *) xcalloc (256 / 32, sizeof (uint32_t));

      /* We only set CLASS_B for the bits in the ISO C classes, not
	 the user defined classes.  The number should not change but
	 who knows.  */
#define LAST_ISO_C_BIT 11
      if (nr <= LAST_ISO_C_BIT)
	for (idx = 0; idx < 256; ++idx)
	  if (ctype->class256_collection[idx] & _ISbit (nr))
	    ctype->class_b[nr][idx >> 5] |= (uint32_t) 1 << (idx & 0x1f);
    }

  for (nr = 0; nr < ctype->nr_charclass; nr++)
    {
      struct wctype_table *t;

      t = &ctype->class_3level[nr];
      t->p = 4; /* or: 5 */
      t->q = 7; /* or: 6 */
      wctype_table_init (t);

      for (idx = 0; idx < ctype->class_collection_act; ++idx)
	if (ctype->class_collection[idx] & _ISwbit (nr))
	  wctype_table_add (t, ctype->charnames[idx]);

      record_verbose (stderr, _("\
%s: table for class \"%s\": %lu bytes"),
		      "LC_CTYPE", ctype->classnames[nr],
		      (unsigned long int) t->result_size);
    }

  /* Room for table of mappings.  */
  ctype->map_b = (uint32_t **) xmalloc (2 * sizeof (uint32_t *));
  ctype->map32_b = (uint32_t **) xmalloc (ctype->map_collection_nr
					  * sizeof (uint32_t *));
  ctype->map_3level = (struct wctrans_table *)
    xmalloc (ctype->map_collection_nr * sizeof (struct wctrans_table));

  /* Fill in all mappings.  */
  for (idx = 0; idx < 2; ++idx)
    {
      unsigned int idx2;

      /* Allocate table.  */
      ctype->map_b[idx] = (uint32_t *)
	xmalloc ((256 + 128) * sizeof (uint32_t));

      /* Copy values from collection.  */
      for (idx2 = 0; idx2 < 256; ++idx2)
	ctype->map_b[idx][128 + idx2] = ctype->map256_collection[idx][idx2];

      /* Mirror first 127 entries.  We must take care not to map entry
	 -1 because EOF == -1.  */
      for (idx2 = 0; idx2 < 127; ++idx2)
	ctype->map_b[idx][idx2] = ctype->map_b[idx][256 + idx2];

      /* EOF must map to EOF.  */
      ctype->map_b[idx][127] = EOF;
    }

  for (idx = 0; idx < ctype->map_collection_nr; ++idx)
    {
      unsigned int idx2;

      /* Allocate table.  */
      ctype->map32_b[idx] = (uint32_t *) xmalloc (256 * sizeof (uint32_t));

      /* Copy values from collection.  Default is identity mapping.  */
      for (idx2 = 0; idx2 < 256; ++idx2)
	ctype->map32_b[idx][idx2] =
	  (ctype->map_collection[idx][idx2] != 0
	   ? ctype->map_collection[idx][idx2]
	   : idx2);
    }

  for (nr = 0; nr < ctype->map_collection_nr; nr++)
    {
      struct wctrans_table *t;

      t = &ctype->map_3level[nr];
      t->p = 7;
      t->q = 9;
      wctrans_table_init (t);

      for (idx = 0; idx < ctype->map_collection_act[nr]; ++idx)
	if (ctype->map_collection[nr][idx] != 0)
	  wctrans_table_add (t, ctype->charnames[idx],
			     ctype->map_collection[nr][idx]);

      record_verbose (stderr, _("\
%s: table for map \"%s\": %lu bytes"),
		      "LC_CTYPE", ctype->mapnames[nr],
		      (unsigned long int) t->result_size);
    }

  /* Extra array for class and map names.  */
  ctype->class_name_ptr = (uint32_t *) xmalloc (ctype->nr_charclass
						* sizeof (uint32_t));
  ctype->map_name_ptr = (uint32_t *) xmalloc (ctype->map_collection_nr
					      * sizeof (uint32_t));

  ctype->class_offset = _NL_ITEM_INDEX (_NL_CTYPE_EXTRA_MAP_1);
  ctype->map_offset = ctype->class_offset + ctype->nr_charclass;

  /* Array for width information.  Because the expected widths are very
     small (never larger than 2) we use only one single byte.  This
     saves space.
     We put only printable characters in the table.  wcwidth is specified
     to return -1 for non-printable characters.  Doing the check here
     saves a run-time check.
     But we put L'\0' in the table.  This again saves a run-time check.  */
  {
    struct wcwidth_table *t;

    t = &ctype->width;
    t->p = 7;
    t->q = 9;
    wcwidth_table_init (t);

    /* First set all the printable characters of the character set to
       the default width.  */
    curs = NULL;
    while (iterate_table (&charmap->char_table, &curs, &key, &len, &vdata) == 0)
      {
	struct charseq *data = (struct charseq *) vdata;

	if (data->ucs4 == UNINITIALIZED_CHAR_VALUE)
	  data->ucs4 = repertoire_find_value (ctype->repertoire,
					      data->name, len);

	if (data->ucs4 != ILLEGAL_CHAR_VALUE)
	  {
	    uint32_t *class_bits =
	      find_idx (ctype, &ctype->class_collection, NULL,
			&ctype->class_collection_act, data->ucs4);

	    if (class_bits != NULL && (*class_bits & BITw (tok_print)))
	      wcwidth_table_add (t, data->ucs4, charmap->width_default);
	  }
      }

    /* Now add the explicitly specified widths.  */
    if (charmap->width_rules != NULL)
      for (size_t cnt = 0; cnt < charmap->nwidth_rules; ++cnt)
        {
          unsigned char bytes[charmap->mb_cur_max];
          int nbytes = charmap->width_rules[cnt].from->nbytes;

          /* We have the range of character for which the width is
             specified described using byte sequences of the multibyte
             charset.  We have to convert this to UCS4 now.  And we
             cannot simply convert the beginning and the end of the
             sequence, we have to iterate over the byte sequence and
             convert it for every single character.  */
          memcpy (bytes, charmap->width_rules[cnt].from->bytes, nbytes);

          while (nbytes < charmap->width_rules[cnt].to->nbytes
                 || memcmp (bytes, charmap->width_rules[cnt].to->bytes,
                            nbytes) <= 0)
            {
              /* Find the UCS value for `bytes'.  */
              int inner;
              uint32_t wch;
              struct charseq *seq =
                charmap_find_symbol (charmap, (char *) bytes, nbytes);

              if (seq == NULL)
                wch = ILLEGAL_CHAR_VALUE;
              else if (seq->ucs4 != UNINITIALIZED_CHAR_VALUE)
                wch = seq->ucs4;
              else
                wch = repertoire_find_value (ctype->repertoire, seq->name,
                                             strlen (seq->name));

              if (wch != ILLEGAL_CHAR_VALUE)
                {
                  /* Store the value.  */
                  uint32_t *class_bits =
                    find_idx (ctype, &ctype->class_collection, NULL,
                              &ctype->class_collection_act, wch);

                  if (class_bits != NULL && (*class_bits & BITw (tok_print)))
                    wcwidth_table_add (t, wch,
                                       charmap->width_rules[cnt].width);
                }

              /* "Increment" the bytes sequence.  */
              inner = nbytes - 1;
              while (inner >= 0 && bytes[inner] == 0xff)
                --inner;

              if (inner < 0)
                {
                  /* We have to extend the byte sequence.  */
                  if (nbytes >= charmap->width_rules[cnt].to->nbytes)
                    break;

                  bytes[0] = 1;
                  memset (&bytes[1], 0, nbytes);
                  ++nbytes;
                }
              else
                {
                  ++bytes[inner];
                  while (++inner < nbytes)
                    bytes[inner] = 0;
                }
            }
        }

    /* Set the width of L'\0' to 0.  */
    wcwidth_table_add (t, 0, 0);

    record_verbose (stderr, _("%s: table for width: %lu bytes"),
		    "LC_CTYPE", (unsigned long int) t->result_size);
  }

  /* Set MB_CUR_MAX.  */
  ctype->mb_cur_max = charmap->mb_cur_max;

  /* Now determine the table for the transliteration information.

     XXX It is not yet clear to me whether it is worth implementing a
     complicated algorithm which uses a hash table to locate the entries.
     For now I'll use a simple array which can be searching using binary
     search.  */
  if (ctype->translit_include != NULL)
    /* Traverse the locales mentioned in the `include' statements in a
       depth-first way and fold in their transliteration information.  */
    translit_flatten (ctype, charmap, &ctype->translit);

  if (ctype->translit != NULL)
    {
      /* First count how many entries we have.  This is the upper limit
	 since some entries from the included files might be overwritten.  */
      size_t number = 0;
      struct translit_t *runp = ctype->translit;
      struct translit_t **sorted;
      size_t from_len, to_len;

      while (runp != NULL)
	{
	  ++number;
	  runp = runp->next;
	}

      /* Next we allocate an array large enough and fill in the values.  */
      sorted = (struct translit_t **) alloca (number
					      * sizeof (struct translit_t **));
      runp = ctype->translit;
      number = 0;
      do
	{
	  /* Search for the place where to insert this string.
	     XXX Better use a real sorting algorithm later.  */
	  size_t idx = 0;
	  int replace = 0;

	  while (idx < number)
	    {
	      int res = wcscmp ((const wchar_t *) sorted[idx]->from,
				(const wchar_t *) runp->from);
	      if (res == 0)
		{
		  replace = 1;
		  break;
		}
	      if (res > 0)
		break;
	      ++idx;
	    }

	  if (replace)
	    sorted[idx] = runp;
	  else
	    {
	      memmove (&sorted[idx + 1], &sorted[idx],
		       (number - idx) * sizeof (struct translit_t *));
	      sorted[idx] = runp;
	      ++number;
	    }

	  runp = runp->next;
	}
      while (runp != NULL);

      /* The next step is putting all the possible transliteration
	 strings in one memory block so that we can write it out.
	 We need several different blocks:
	 - index to the from-string array
	 - from-string array
	 - index to the to-string array
	 - to-string array.
      */
      from_len = to_len = 0;
      for (size_t cnt = 0; cnt < number; ++cnt)
	{
	  struct translit_to_t *srunp;
	  from_len += wcslen ((const wchar_t *) sorted[cnt]->from) + 1;
	  srunp = sorted[cnt]->to;
	  while (srunp != NULL)
	    {
	      to_len += wcslen ((const wchar_t *) srunp->str) + 1;
	      srunp = srunp->next;
	    }
	  /* Plus one for the extra NUL character marking the end of
	     the list for the current entry.  */
	  ++to_len;
	}

      /* We can allocate the arrays for the results.  */
      ctype->translit_from_idx = xmalloc (number * sizeof (uint32_t));
      ctype->translit_from_tbl = xmalloc (from_len * sizeof (uint32_t));
      ctype->translit_to_idx = xmalloc (number * sizeof (uint32_t));
      ctype->translit_to_tbl = xmalloc (to_len * sizeof (uint32_t));

      from_len = 0;
      to_len = 0;
      for (size_t cnt = 0; cnt < number; ++cnt)
	{
	  size_t len;
	  struct translit_to_t *srunp;

	  ctype->translit_from_idx[cnt] = from_len;
	  ctype->translit_to_idx[cnt] = to_len;

	  len = wcslen ((const wchar_t *) sorted[cnt]->from) + 1;
	  wmemcpy ((wchar_t *) &ctype->translit_from_tbl[from_len],
		   (const wchar_t *) sorted[cnt]->from, len);
	  from_len += len;

	  ctype->translit_to_idx[cnt] = to_len;
	  srunp = sorted[cnt]->to;
	  while (srunp != NULL)
	    {
	      len = wcslen ((const wchar_t *) srunp->str) + 1;
	      wmemcpy ((wchar_t *) &ctype->translit_to_tbl[to_len],
		       (const wchar_t *) srunp->str, len);
	      to_len += len;
	      srunp = srunp->next;
	    }
	  ctype->translit_to_tbl[to_len++] = L'\0';
	}

      /* Store the information about the length.  */
      ctype->translit_idx_size = number;
      ctype->translit_from_tbl_size = from_len * sizeof (uint32_t);
      ctype->translit_to_tbl_size = to_len * sizeof (uint32_t);
    }
  else
    {
      ctype->translit_from_idx = no_str;
      ctype->translit_from_tbl = no_str;
      ctype->translit_to_tbl = no_str;
      ctype->translit_idx_size = 0;
      ctype->translit_from_tbl_size = 0;
      ctype->translit_to_tbl_size = 0;
    }
}
