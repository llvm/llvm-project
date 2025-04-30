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

#include <errno.h>
#include <stdlib.h>
#include <wchar.h>
#include <stdint.h>
#include <sys/param.h>

#include "localedef.h"
#include "charmap.h"
#include "localeinfo.h"
#include "linereader.h"
#include "locfile.h"
#include "elem-hash.h"

/* Uncomment the following line in the production version.  */
/* #define NDEBUG 1 */
#include <assert.h>

#define obstack_chunk_alloc malloc
#define obstack_chunk_free free

static inline void
__attribute ((always_inline))
obstack_int32_grow (struct obstack *obstack, int32_t data)
{
  assert (LOCFILE_ALIGNED_P (obstack_object_size (obstack)));
  data = maybe_swap_uint32 (data);
  if (sizeof (int32_t) == sizeof (int))
    obstack_int_grow (obstack, data);
  else
    obstack_grow (obstack, &data, sizeof (int32_t));
}

static inline void
__attribute ((always_inline))
obstack_int32_grow_fast (struct obstack *obstack, int32_t data)
{
  assert (LOCFILE_ALIGNED_P (obstack_object_size (obstack)));
  data = maybe_swap_uint32 (data);
  if (sizeof (int32_t) == sizeof (int))
    obstack_int_grow_fast (obstack, data);
  else
    obstack_grow (obstack, &data, sizeof (int32_t));
}

/* Forward declaration.  */
struct element_t;

/* Data type for list of strings.  */
struct section_list
{
  /* Successor in the known_sections list.  */
  struct section_list *def_next;
  /* Successor in the sections list.  */
  struct section_list *next;
  /* Name of the section.  */
  const char *name;
  /* First element of this section.  */
  struct element_t *first;
  /* Last element of this section.  */
  struct element_t *last;
  /* These are the rules for this section.  */
  enum coll_sort_rule *rules;
  /* Index of the rule set in the appropriate section of the output file.  */
  int ruleidx;
};

struct element_t;

struct element_list_t
{
  /* Number of elements.  */
  int cnt;

  struct element_t **w;
};

/* Data type for collating element.  */
struct element_t
{
  const char *name;

  const char *mbs;
  size_t nmbs;
  const uint32_t *wcs;
  size_t nwcs;
  int *mborder;
  int wcorder;

  /* The following is a bit mask which bits are set if this element is
     used in the appropriate level.  Interesting for the singlebyte
     weight computation.

     XXX The type here restricts the number of levels to 32.  It could
     be changed if necessary but I doubt this is necessary.  */
  unsigned int used_in_level;

  struct element_list_t *weights;

  /* Nonzero if this is a real character definition.  */
  int is_character;

  /* Order of the character in the sequence.  This information will
     be used in range expressions.  */
  int mbseqorder;
  int wcseqorder;

  /* Where does the definition come from.  */
  const char *file;
  size_t line;

  /* Which section does this belong to.  */
  struct section_list *section;

  /* Predecessor and successor in the order list.  */
  struct element_t *last;
  struct element_t *next;

  /* Next element in multibyte output list.  */
  struct element_t *mbnext;
  struct element_t *mblast;

  /* Next element in wide character output list.  */
  struct element_t *wcnext;
  struct element_t *wclast;
};

/* Special element value.  */
#define ELEMENT_ELLIPSIS2	((struct element_t *) 1)
#define ELEMENT_ELLIPSIS3	((struct element_t *) 2)
#define ELEMENT_ELLIPSIS4	((struct element_t *) 3)

/* Data type for collating symbol.  */
struct symbol_t
{
  const char *name;

  /* Point to place in the order list.  */
  struct element_t *order;

  /* Where does the definition come from.  */
  const char *file;
  size_t line;
};

/* Sparse table of struct element_t *.  */
#define TABLE wchead_table
#define ELEMENT struct element_t *
#define DEFAULT NULL
#define ITERATE
#define NO_ADD_LOCALE
#include "3level.h"

/* Sparse table of int32_t.  */
#define TABLE collidx_table
#define ELEMENT int32_t
#define DEFAULT 0
#include "3level.h"

/* Sparse table of uint32_t.  */
#define TABLE collseq_table
#define ELEMENT uint32_t
#define DEFAULT ~((uint32_t) 0)
#include "3level.h"


/* Simple name list for the preprocessor.  */
struct name_list
{
  struct name_list *next;
  char str[0];
};


/* The real definition of the struct for the LC_COLLATE locale.  */
struct locale_collate_t
{
  int col_weight_max;
  int cur_weight_max;

  /* List of known scripts.  */
  struct section_list *known_sections;
  /* List of used sections.  */
  struct section_list *sections;
  /* Current section using definition.  */
  struct section_list *current_section;
  /* There always can be an unnamed section.  */
  struct section_list unnamed_section;
  /* Flag whether the unnamed section has been defined.  */
  bool unnamed_section_defined;
  /* To make handling of errors easier we have another section.  */
  struct section_list error_section;
  /* Sometimes we are defining the values for collating symbols before
     the first actual section.  */
  struct section_list symbol_section;

  /* Start of the order list.  */
  struct element_t *start;

  /* The undefined element.  */
  struct element_t undefined;

  /* This is the cursor for `reorder_after' insertions.  */
  struct element_t *cursor;

  /* This value is used when handling ellipsis.  */
  struct element_t ellipsis_weight;

  /* Known collating elements.  */
  hash_table elem_table;

  /* Known collating symbols.  */
  hash_table sym_table;

  /* Known collation sequences.  */
  hash_table seq_table;

  struct obstack mempool;

  /* The LC_COLLATE category is a bit special as it is sometimes possible
     that the definitions from more than one input file contains information.
     Therefore we keep all relevant input in a list.  */
  struct locale_collate_t *next;

  /* Arrays with heads of the list for each of the leading bytes in
     the multibyte sequences.  */
  struct element_t *mbheads[256];

  /* Arrays with heads of the list for each of the leading bytes in
     the multibyte sequences.  */
  struct wchead_table wcheads;

  /* The arrays with the collation sequence order.  */
  unsigned char mbseqorder[256];
  struct collseq_table wcseqorder;

  /* State of the preprocessor.  */
  enum
    {
      else_none = 0,
      else_ignore,
      else_seen
    }
    else_action;
};


/* We have a few global variables which are used for reading all
   LC_COLLATE category descriptions in all files.  */
static uint32_t nrules;

/* List of defined preprocessor symbols.  */
static struct name_list *defined;


/* We need UTF-8 encoding of numbers.  */
static inline int
__attribute ((always_inline))
utf8_encode (char *buf, int val)
{
  int retval;

  if (val < 0x80)
    {
      *buf++ = (char) val;
      retval = 1;
    }
  else
    {
      int step;

      for (step = 2; step < 6; ++step)
	if ((val & (~(uint32_t)0 << (5 * step + 1))) == 0)
	  break;
      retval = step;

      *buf = (unsigned char) (~0xff >> step);
      --step;
      do
	{
	  buf[step] = 0x80 | (val & 0x3f);
	  val >>= 6;
	}
      while (--step > 0);
      *buf |= val;
    }

  return retval;
}


static struct section_list *
make_seclist_elem (struct locale_collate_t *collate, const char *string,
		   struct section_list *next)
{
  struct section_list *newp;

  newp = (struct section_list *) obstack_alloc (&collate->mempool,
						sizeof (*newp));
  newp->next = next;
  newp->name = string;
  newp->first = NULL;
  newp->last = NULL;

  return newp;
}


static struct element_t *
new_element (struct locale_collate_t *collate, const char *mbs, size_t mbslen,
	     const uint32_t *wcs, const char *name, size_t namelen,
	     int is_character)
{
  struct element_t *newp;

  newp = (struct element_t *) obstack_alloc (&collate->mempool,
					     sizeof (*newp));
  newp->name = name == NULL ? NULL : obstack_copy0 (&collate->mempool,
						    name, namelen);
  if (mbs != NULL)
    {
      newp->mbs = obstack_copy0 (&collate->mempool, mbs, mbslen);
      newp->nmbs = mbslen;
    }
  else
    {
      newp->mbs = NULL;
      newp->nmbs = 0;
    }
  if (wcs != NULL)
    {
      size_t nwcs = wcslen ((wchar_t *) wcs);
      uint32_t zero = 0;
      /* Handle <U0000> as a single character.  */
      if (nwcs == 0)
	nwcs = 1;
      obstack_grow (&collate->mempool, wcs, nwcs * sizeof (uint32_t));
      obstack_grow (&collate->mempool, &zero, sizeof (uint32_t));
      newp->wcs = (uint32_t *) obstack_finish (&collate->mempool);
      newp->nwcs = nwcs;
    }
  else
    {
      newp->wcs = NULL;
      newp->nwcs = 0;
    }
  newp->mborder = NULL;
  newp->wcorder = 0;
  newp->used_in_level = 0;
  newp->is_character = is_character;

  /* Will be assigned later.  XXX  */
  newp->mbseqorder = 0;
  newp->wcseqorder = 0;

  /* Will be allocated later.  */
  newp->weights = NULL;

  newp->file = NULL;
  newp->line = 0;

  newp->section = collate->current_section;

  newp->last = NULL;
  newp->next = NULL;

  newp->mbnext = NULL;
  newp->mblast = NULL;

  newp->wcnext = NULL;
  newp->wclast = NULL;

  return newp;
}


static struct symbol_t *
new_symbol (struct locale_collate_t *collate, const char *name, size_t len)
{
  struct symbol_t *newp;

  newp = (struct symbol_t *) obstack_alloc (&collate->mempool, sizeof (*newp));

  newp->name = obstack_copy0 (&collate->mempool, name, len);
  newp->order = NULL;

  newp->file = NULL;
  newp->line = 0;

  return newp;
}


/* Test whether this name is already defined somewhere.  */
static int
check_duplicate (struct linereader *ldfile, struct locale_collate_t *collate,
		 const struct charmap_t *charmap,
		 struct repertoire_t *repertoire, const char *symbol,
		 size_t symbol_len)
{
  void *ignore = NULL;

  if (find_entry (&charmap->char_table, symbol, symbol_len, &ignore) == 0)
    {
      lr_error (ldfile, _("`%.*s' already defined in charmap"),
		(int) symbol_len, symbol);
      return 1;
    }

  if (repertoire != NULL
      && (find_entry (&repertoire->char_table, symbol, symbol_len, &ignore)
	  == 0))
    {
      lr_error (ldfile, _("`%.*s' already defined in repertoire"),
		(int) symbol_len, symbol);
      return 1;
    }

  if (find_entry (&collate->sym_table, symbol, symbol_len, &ignore) == 0)
    {
      lr_error (ldfile, _("`%.*s' already defined as collating symbol"),
		(int) symbol_len, symbol);
      return 1;
    }

  if (find_entry (&collate->elem_table, symbol, symbol_len, &ignore) == 0)
    {
      lr_error (ldfile, _("`%.*s' already defined as collating element"),
		(int) symbol_len, symbol);
      return 1;
    }

  return 0;
}


/* Read the direction specification.  */
static void
read_directions (struct linereader *ldfile, struct token *arg,
		 const struct charmap_t *charmap,
		 struct repertoire_t *repertoire, struct localedef_t *result)
{
  int cnt = 0;
  int max = nrules ?: 10;
  enum coll_sort_rule *rules = calloc (max, sizeof (*rules));
  int warned = 0;
  struct locale_collate_t *collate = result->categories[LC_COLLATE].collate;

  while (1)
    {
      int valid = 0;

      if (arg->tok == tok_forward)
	{
	  if (rules[cnt] & sort_backward)
	    {
	      if (! warned)
		{
		  lr_error (ldfile, _("\
%s: `forward' and `backward' are mutually excluding each other"),
			    "LC_COLLATE");
		  warned = 1;
		}
	    }
	  else if (rules[cnt] & sort_forward)
	    {
	      if (! warned)
		{
		  lr_error (ldfile, _("\
%s: `%s' mentioned more than once in definition of weight %d"),
			    "LC_COLLATE", "forward", cnt + 1);
		}
	    }
	  else
	    rules[cnt] |= sort_forward;

	  valid = 1;
	}
      else if (arg->tok == tok_backward)
	{
	  if (rules[cnt] & sort_forward)
	    {
	      if (! warned)
		{
		  lr_error (ldfile, _("\
%s: `forward' and `backward' are mutually excluding each other"),
			    "LC_COLLATE");
		  warned = 1;
		}
	    }
	  else if (rules[cnt] & sort_backward)
	    {
	      if (! warned)
		{
		  lr_error (ldfile, _("\
%s: `%s' mentioned more than once in definition of weight %d"),
			    "LC_COLLATE", "backward", cnt + 1);
		}
	    }
	  else
	    rules[cnt] |= sort_backward;

	  valid = 1;
	}
      else if (arg->tok == tok_position)
	{
	  if (rules[cnt] & sort_position)
	    {
	      if (! warned)
		{
		  lr_error (ldfile, _("\
%s: `%s' mentioned more than once in definition of weight %d"),
			    "LC_COLLATE", "position", cnt + 1);
		}
	    }
	  else
	    rules[cnt] |= sort_position;

	  valid = 1;
	}

      if (valid)
	arg = lr_token (ldfile, charmap, result, repertoire, verbose);

      if (arg->tok == tok_eof || arg->tok == tok_eol || arg->tok == tok_comma
	  || arg->tok == tok_semicolon)
	{
	  if (! valid && ! warned)
	    {
	      lr_error (ldfile, _("%s: syntax error"), "LC_COLLATE");
	      warned = 1;
	    }

	  /* See whether we have to increment the counter.  */
	  if (arg->tok != tok_comma && rules[cnt] != 0)
	    {
	      /* Add the default `forward' if we have seen only `position'.  */
	      if (rules[cnt] == sort_position)
		rules[cnt] = sort_position | sort_forward;

	      ++cnt;
	    }

	  if (arg->tok == tok_eof || arg->tok == tok_eol)
	    /* End of line or file, so we exit the loop.  */
	    break;

	  if (nrules == 0)
	    {
	      /* See whether we have enough room in the array.  */
	      if (cnt == max)
		{
		  max += 10;
		  rules = (enum coll_sort_rule *) xrealloc (rules,
							    max
							    * sizeof (*rules));
		  memset (&rules[cnt], '\0', (max - cnt) * sizeof (*rules));
		}
	    }
	  else
	    {
	      if (cnt == nrules)
		{
		  /* There must not be any more rule.  */
		  if (! warned)
		    {
		      lr_error (ldfile, _("\
%s: too many rules; first entry only had %d"),
				"LC_COLLATE", nrules);
		      warned = 1;
		    }

		  lr_ignore_rest (ldfile, 0);
		  break;
		}
	    }
	}
      else
	{
	  if (! warned)
	    {
	      lr_error (ldfile, _("%s: syntax error"), "LC_COLLATE");
	      warned = 1;
	    }
	}

      arg = lr_token (ldfile, charmap, result, repertoire, verbose);
    }

  if (nrules == 0)
    {
      /* Now we know how many rules we have.  */
      nrules = cnt;
      rules = (enum coll_sort_rule *) xrealloc (rules,
						nrules * sizeof (*rules));
    }
  else
    {
      if (cnt < nrules)
	{
	  /* Not enough rules in this specification.  */
	  if (! warned)
	    lr_error (ldfile, _("%s: not enough sorting rules"), "LC_COLLATE");

	  do
	    rules[cnt] = sort_forward;
	  while (++cnt < nrules);
	}
    }

  collate->current_section->rules = rules;
}


static struct element_t *
find_element (struct linereader *ldfile, struct locale_collate_t *collate,
	      const char *str, size_t len)
{
  void *result = NULL;

  /* Search for the entries among the collation sequences already define.  */
  if (find_entry (&collate->seq_table, str, len, &result) != 0)
    {
      /* Nope, not define yet.  So we see whether it is a
	 collation symbol.  */
      void *ptr;

      if (find_entry (&collate->sym_table, str, len, &ptr) == 0)
	{
	  /* It's a collation symbol.  */
	  struct symbol_t *sym = (struct symbol_t *) ptr;
	  result = sym->order;

	  if (result == NULL)
	    result = sym->order = new_element (collate, NULL, 0, NULL,
					       NULL, 0, 0);
	}
      else if (find_entry (&collate->elem_table, str, len, &result) != 0)
	{
	  /* It's also no collation element.  So it is a character
	     element defined later.  */
	  result = new_element (collate, NULL, 0, NULL, str, len, 1);
	  /* Insert it into the sequence table.  */
	  insert_entry (&collate->seq_table, str, len, result);
	}
    }

  return (struct element_t *) result;
}


static void
unlink_element (struct locale_collate_t *collate)
{
  if (collate->cursor == collate->start)
    {
      assert (collate->cursor->next == NULL);
      assert (collate->cursor->last == NULL);
      collate->cursor = NULL;
    }
  else
    {
      if (collate->cursor->next != NULL)
	collate->cursor->next->last = collate->cursor->last;
      if (collate->cursor->last != NULL)
	collate->cursor->last->next = collate->cursor->next;
      collate->cursor = collate->cursor->last;
    }
}


static void
insert_weights (struct linereader *ldfile, struct element_t *elem,
		const struct charmap_t *charmap,
		struct repertoire_t *repertoire, struct localedef_t *result,
		enum token_t ellipsis)
{
  int weight_cnt;
  struct token *arg;
  struct locale_collate_t *collate = result->categories[LC_COLLATE].collate;

  /* Initialize all the fields.  */
  elem->file = ldfile->fname;
  elem->line = ldfile->lineno;

  elem->last = collate->cursor;
  elem->next = collate->cursor ? collate->cursor->next : NULL;
  if (collate->cursor != NULL && collate->cursor->next != NULL)
    collate->cursor->next->last = elem;
  if (collate->cursor != NULL)
    collate->cursor->next = elem;
  if (collate->start == NULL)
    {
      assert (collate->cursor == NULL);
      collate->start = elem;
    }

  elem->section = collate->current_section;

  if (collate->current_section->first == NULL)
    collate->current_section->first = elem;
  if (collate->current_section->last == collate->cursor)
    collate->current_section->last = elem;

  collate->cursor = elem;

  elem->weights = (struct element_list_t *)
    obstack_alloc (&collate->mempool, nrules * sizeof (struct element_list_t));
  memset (elem->weights, '\0', nrules * sizeof (struct element_list_t));

  weight_cnt = 0;

  arg = lr_token (ldfile, charmap, result, repertoire, verbose);
  do
    {
      if (arg->tok == tok_eof || arg->tok == tok_eol)
	break;

      if (arg->tok == tok_ignore)
	{
	  /* The weight for this level has to be ignored.  We use the
	     null pointer to indicate this.  */
	  elem->weights[weight_cnt].w = (struct element_t **)
	    obstack_alloc (&collate->mempool, sizeof (struct element_t *));
	  elem->weights[weight_cnt].w[0] = NULL;
	  elem->weights[weight_cnt].cnt = 1;
	}
      else if (arg->tok == tok_bsymbol || arg->tok == tok_ucs4)
	{
	  char ucs4str[10];
	  struct element_t *val;
	  char *symstr;
	  size_t symlen;

	  if (arg->tok == tok_bsymbol)
	    {
	      symstr = arg->val.str.startmb;
	      symlen = arg->val.str.lenmb;
	    }
	  else
	    {
	      snprintf (ucs4str, sizeof (ucs4str), "U%08X", arg->val.ucs4);
	      symstr = ucs4str;
	      symlen = 9;
	    }

	  val = find_element (ldfile, collate, symstr, symlen);
	  if (val == NULL)
	    break;

	  elem->weights[weight_cnt].w = (struct element_t **)
	    obstack_alloc (&collate->mempool, sizeof (struct element_t *));
	  elem->weights[weight_cnt].w[0] = val;
	  elem->weights[weight_cnt].cnt = 1;
	}
      else if (arg->tok == tok_string)
	{
	  /* Split the string up in the individual characters and put
	     the element definitions in the list.  */
	  const char *cp = arg->val.str.startmb;
	  int cnt = 0;
	  struct element_t *charelem;
	  struct element_t **weights = NULL;
	  int max = 0;

	  if (*cp == '\0')
	    {
	      lr_error (ldfile, _("%s: empty weight string not allowed"),
			"LC_COLLATE");
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  do
	    {
	      if (*cp == '<')
		{
		  /* Ahh, it's a bsymbol or an UCS4 value.  If it's
		     the latter we have to unify the name.  */
		  const char *startp = ++cp;
		  size_t len;

		  while (*cp != '>')
		    {
		      if (*cp == ldfile->escape_char)
			++cp;
		      if (*cp == '\0')
			/* It's a syntax error.  */
			goto syntax;

		      ++cp;
		    }

		  if (cp - startp == 5 && startp[0] == 'U'
		      && isxdigit (startp[1]) && isxdigit (startp[2])
		      && isxdigit (startp[3]) && isxdigit (startp[4]))
		    {
		      unsigned int ucs4 = strtoul (startp + 1, NULL, 16);
		      char *newstr;

		      newstr = (char *) xmalloc (10);
		      snprintf (newstr, 10, "U%08X", ucs4);
		      startp = newstr;

		      len = 9;
		    }
		  else
		    len = cp - startp;

		  charelem = find_element (ldfile, collate, startp, len);
		  ++cp;
		}
	      else
		{
		  /* People really shouldn't use characters directly in
		     the string.  Especially since it's not really clear
		     what this means.  We interpret all characters in the
		     string as if that would be bsymbols.  Otherwise we
		     would have to match back to bsymbols somehow and this
		     is normally not what people normally expect.  */
		  charelem = find_element (ldfile, collate, cp++, 1);
		}

	      if (charelem == NULL)
		{
		  /* We ignore the rest of the line.  */
		  lr_ignore_rest (ldfile, 0);
		  break;
		}

	      /* Add the pointer.  */
	      if (cnt >= max)
		{
		  struct element_t **newp;
		  max += 10;
		  newp = (struct element_t **)
		    alloca (max * sizeof (struct element_t *));
		  memcpy (newp, weights, cnt * sizeof (struct element_t *));
		  weights = newp;
		}
	      weights[cnt++] = charelem;
	    }
	  while (*cp != '\0');

	  /* Now store the information.  */
	  elem->weights[weight_cnt].w = (struct element_t **)
	    obstack_alloc (&collate->mempool,
			   cnt * sizeof (struct element_t *));
	  memcpy (elem->weights[weight_cnt].w, weights,
		  cnt * sizeof (struct element_t *));
	  elem->weights[weight_cnt].cnt = cnt;

	  /* We don't need the string anymore.  */
	  free (arg->val.str.startmb);
	}
      else if (ellipsis != tok_none
	       && (arg->tok == tok_ellipsis2
		   || arg->tok == tok_ellipsis3
		   || arg->tok == tok_ellipsis4))
	{
	  /* It must be the same ellipsis as used in the initial column.  */
	  if (arg->tok != ellipsis)
	    lr_error (ldfile, _("\
%s: weights must use the same ellipsis symbol as the name"),
		      "LC_COLLATE");

	  /* The weight for this level will depend on the element
	     iterating over the range.  Put a placeholder.  */
	  elem->weights[weight_cnt].w = (struct element_t **)
	    obstack_alloc (&collate->mempool, sizeof (struct element_t *));
	  elem->weights[weight_cnt].w[0] = ELEMENT_ELLIPSIS2;
	  elem->weights[weight_cnt].cnt = 1;
	}
      else
	{
	syntax:
	  /* It's a syntax error.  */
	  lr_error (ldfile, _("%s: syntax error"), "LC_COLLATE");
	  lr_ignore_rest (ldfile, 0);
	  break;
	}

      arg = lr_token (ldfile, charmap, result, repertoire, verbose);
      /* This better should be the end of the line or a semicolon.  */
      if (arg->tok == tok_semicolon)
	/* OK, ignore this and read the next token.  */
	arg = lr_token (ldfile, charmap, result, repertoire, verbose);
      else if (arg->tok != tok_eof && arg->tok != tok_eol)
	{
	  /* It's a syntax error.  */
	  lr_error (ldfile, _("%s: syntax error"), "LC_COLLATE");
	  lr_ignore_rest (ldfile, 0);
	  break;
	}
    }
  while (++weight_cnt < nrules);

  if (weight_cnt < nrules)
    {
      /* This means the rest of the line uses the current element as
	 the weight.  */
      do
	{
	  elem->weights[weight_cnt].w = (struct element_t **)
	    obstack_alloc (&collate->mempool, sizeof (struct element_t *));
	  if (ellipsis == tok_none)
	    elem->weights[weight_cnt].w[0] = elem;
	  else
	    elem->weights[weight_cnt].w[0] = ELEMENT_ELLIPSIS2;
	  elem->weights[weight_cnt].cnt = 1;
	}
      while (++weight_cnt < nrules);
    }
  else
    {
      if (arg->tok == tok_ignore || arg->tok == tok_bsymbol)
	{
	  /* Too many rule values.  */
	  lr_error (ldfile, _("%s: too many values"), "LC_COLLATE");
	  lr_ignore_rest (ldfile, 0);
	}
      else
	lr_ignore_rest (ldfile, arg->tok != tok_eol && arg->tok != tok_eof);
    }
}


static int
insert_value (struct linereader *ldfile, const char *symstr, size_t symlen,
	      const struct charmap_t *charmap, struct repertoire_t *repertoire,
	      struct localedef_t *result)
{
  /* First find out what kind of symbol this is.  */
  struct charseq *seq;
  uint32_t wc;
  struct element_t *elem = NULL;
  struct locale_collate_t *collate = result->categories[LC_COLLATE].collate;

  /* Try to find the character in the charmap.  */
  seq = charmap_find_value (charmap, symstr, symlen);

  /* Determine the wide character.  */
  if (seq == NULL || seq->ucs4 == UNINITIALIZED_CHAR_VALUE)
    {
      wc = repertoire_find_value (repertoire, symstr, symlen);
      if (seq != NULL)
	seq->ucs4 = wc;
    }
  else
    wc = seq->ucs4;

  if (wc == ILLEGAL_CHAR_VALUE && seq == NULL)
    {
      /* It's no character, so look through the collation elements and
	 symbol list.  */
      void *ptr = elem;
      if (find_entry (&collate->elem_table, symstr, symlen, &ptr) != 0)
	{
	  void *result;
	  struct symbol_t *sym = NULL;

	  /* It's also collation element.  Therefore it's either a
	     collating symbol or it's a character which is not
	     supported by the character set.  In the later case we
	     simply create a dummy entry.  */
	  if (find_entry (&collate->sym_table, symstr, symlen, &result) == 0)
	    {
	      /* It's a collation symbol.  */
	      sym = (struct symbol_t *) result;

	      elem = sym->order;
	    }

	  if (elem == NULL)
	    {
	      elem = new_element (collate, NULL, 0, NULL, symstr, symlen, 0);

	      if (sym != NULL)
		sym->order = elem;
	      else
		/* Enter a fake element in the sequence table.  This
		   won't cause anything in the output since there is
		   no multibyte or wide character associated with
		   it.  */
		insert_entry (&collate->seq_table, symstr, symlen, elem);
	    }
	}
      else
	/* Copy the result back.  */
	elem = ptr;
    }
  else
    {
      /* Otherwise the symbols stands for a character.  */
      void *ptr = elem;
      if (find_entry (&collate->seq_table, symstr, symlen, &ptr) != 0)
	{
	  uint32_t wcs[2] = { wc, 0 };

	  /* We have to allocate an entry.  */
	  elem = new_element (collate,
			      seq != NULL ? (char *) seq->bytes : NULL,
			      seq != NULL ? seq->nbytes : 0,
			      wc == ILLEGAL_CHAR_VALUE ? NULL : wcs,
			      symstr, symlen, 1);

	  /* And add it to the table.  */
	  if (insert_entry (&collate->seq_table, symstr, symlen, elem) != 0)
	    /* This cannot happen.  */
	    assert (! "Internal error");
	}
      else
	{
	  /* Copy the result back.  */
	  elem = ptr;

	  /* Maybe the character was used before the definition.  In this case
	     we have to insert the byte sequences now.  */
	  if (elem->mbs == NULL && seq != NULL)
	    {
	      elem->mbs = obstack_copy0 (&collate->mempool,
					 seq->bytes, seq->nbytes);
	      elem->nmbs = seq->nbytes;
	    }

	  if (elem->wcs == NULL && wc != ILLEGAL_CHAR_VALUE)
	    {
	      uint32_t wcs[2] = { wc, 0 };

	      elem->wcs = obstack_copy (&collate->mempool, wcs, sizeof (wcs));
	      elem->nwcs = 1;
	    }
	}
    }

  /* Test whether this element is not already in the list.  */
  if (elem->next != NULL || elem == collate->cursor)
    {
      lr_error (ldfile, _("order for `%.*s' already defined at %s:%zu"),
		(int) symlen, symstr, elem->file, elem->line);
      lr_ignore_rest (ldfile, 0);
      return 1;
    }

  insert_weights (ldfile, elem, charmap, repertoire, result, tok_none);

  return 0;
}


static void
handle_ellipsis (struct linereader *ldfile, const char *symstr, size_t symlen,
		 enum token_t ellipsis, const struct charmap_t *charmap,
		 struct repertoire_t *repertoire,
		 struct localedef_t *result)
{
  struct element_t *startp;
  struct element_t *endp;
  struct locale_collate_t *collate = result->categories[LC_COLLATE].collate;

  /* Unlink the entry added for the ellipsis.  */
  unlink_element (collate);
  startp = collate->cursor;

  /* Process and add the end-entry.  */
  if (symstr != NULL
      && insert_value (ldfile, symstr, symlen, charmap, repertoire, result))
    /* Something went wrong with inserting the to-value.  This means
       we cannot process the ellipsis.  */
    return;

  /* Reset the cursor.  */
  collate->cursor = startp;

  /* Now we have to handle many different situations:
     - we have to distinguish between the three different ellipsis forms
     - the is the ellipsis at the beginning, in the middle, or at the end.
  */
  endp = collate->cursor->next;
  assert (symstr == NULL || endp != NULL);

  /* XXX The following is probably very wrong since also collating symbols
     can appear in ranges.  But do we want/can refine the test for that?  */
#if 0
  /* Both, the start and the end symbol, must stand for characters.  */
  if ((startp != NULL && (startp->name == NULL || ! startp->is_character))
      || (endp != NULL && (endp->name == NULL|| ! endp->is_character)))
    {
      lr_error (ldfile, _("\
%s: the start and the end symbol of a range must stand for characters"),
		"LC_COLLATE");
      return;
    }
#endif

  if (ellipsis == tok_ellipsis3)
    {
      /* One requirement we make here: the length of the byte
	 sequences for the first and end character must be the same.
	 This is mainly to prevent unwanted effects and this is often
	 not what is wanted.  */
      size_t len = (startp->mbs != NULL ? startp->nmbs
		    : (endp->mbs != NULL ? endp->nmbs : 0));
      char mbcnt[len + 1];
      char mbend[len + 1];

      /* Well, this should be caught somewhere else already.  Just to
	 make sure.  */
      assert (startp == NULL || startp->wcs == NULL || startp->wcs[1] == 0);
      assert (endp == NULL || endp->wcs == NULL || endp->wcs[1] == 0);

      if (startp != NULL && endp != NULL
	  && startp->mbs != NULL && endp->mbs != NULL
	  && startp->nmbs != endp->nmbs)
	{
	  lr_error (ldfile, _("\
%s: byte sequences of first and last character must have the same length"),
		    "LC_COLLATE");
	  return;
	}

      /* Determine whether we have to generate multibyte sequences.  */
      if ((startp == NULL || startp->mbs != NULL)
	  && (endp == NULL || endp->mbs != NULL))
	{
	  int cnt;
	  int ret;

	  /* Prepare the beginning byte sequence.  This is either from the
	     beginning byte sequence or it is all nulls if it was an
	     initial ellipsis.  */
	  if (startp == NULL || startp->mbs == NULL)
	    memset (mbcnt, '\0', len);
	  else
	    {
	      memcpy (mbcnt, startp->mbs, len);

	      /* And increment it so that the value is the first one we will
		 try to insert.  */
	      for (cnt = len - 1; cnt >= 0; --cnt)
		if (++mbcnt[cnt] != '\0')
		  break;
	    }
	  mbcnt[len] = '\0';

	  /* And the end sequence.  */
	  if (endp == NULL || endp->mbs == NULL)
	    memset (mbend, '\0', len);
	  else
	    memcpy (mbend, endp->mbs, len);
	  mbend[len] = '\0';

	  /* Test whether we have a correct range.  */
	  ret = memcmp (mbcnt, mbend, len);
	  if (ret >= 0)
	    {
	      if (ret > 0)
		lr_error (ldfile, _("%s: byte sequence of first character of \
range is not lower than that of the last character"), "LC_COLLATE");
	      return;
	    }

	  /* Generate the byte sequences data.  */
	  while (1)
	    {
	      struct charseq *seq;

	      /* Quite a bit of work ahead.  We have to find the character
		 definition for the byte sequence and then determine the
		 wide character belonging to it.  */
	      seq = charmap_find_symbol (charmap, mbcnt, len);
	      if (seq != NULL)
		{
		  struct element_t *elem;
		  size_t namelen;

		  /* I don't think this can ever happen.  */
		  assert (seq->name != NULL);
		  namelen = strlen (seq->name);

		  if (seq->ucs4 == UNINITIALIZED_CHAR_VALUE)
		    seq->ucs4 = repertoire_find_value (repertoire, seq->name,
						       namelen);

		  /* Now we are ready to insert the new value in the
		     sequence.  Find out whether the element is
		     already known.  */
		  void *ptr;
		  if (find_entry (&collate->seq_table, seq->name, namelen,
				  &ptr) != 0)
		    {
		      uint32_t wcs[2] = { seq->ucs4, 0 };

		      /* We have to allocate an entry.  */
		      elem = new_element (collate, mbcnt, len,
					  seq->ucs4 == ILLEGAL_CHAR_VALUE
					  ? NULL : wcs, seq->name,
					  namelen, 1);

		      /* And add it to the table.  */
		      if (insert_entry (&collate->seq_table, seq->name,
					namelen, elem) != 0)
			/* This cannot happen.  */
			assert (! "Internal error");
		    }
		  else
		    /* Copy the result.  */
		    elem = ptr;

		  /* Test whether this element is not already in the list.  */
		  if (elem->next != NULL || (collate->cursor != NULL
					     && elem->next == collate->cursor))
		    {
		      lr_error (ldfile, _("\
order for `%.*s' already defined at %s:%zu"),
				(int) namelen, seq->name,
				elem->file, elem->line);
		      goto increment;
		    }

		  /* Enqueue the new element.  */
		  elem->last = collate->cursor;
		  if (collate->cursor == NULL)
		    elem->next = NULL;
		  else
		    {
		      elem->next = collate->cursor->next;
		      elem->last->next = elem;
		      if (elem->next != NULL)
			elem->next->last = elem;
		    }
		  if (collate->start == NULL)
		    {
		      assert (collate->cursor == NULL);
		      collate->start = elem;
		    }
		  collate->cursor = elem;

		 /* Add the weight value.  We take them from the
		    `ellipsis_weights' member of `collate'.  */
		  elem->weights = (struct element_list_t *)
		    obstack_alloc (&collate->mempool,
				   nrules * sizeof (struct element_list_t));
		  for (cnt = 0; cnt < nrules; ++cnt)
		    if (collate->ellipsis_weight.weights[cnt].cnt == 1
			&& (collate->ellipsis_weight.weights[cnt].w[0]
			    == ELEMENT_ELLIPSIS2))
		      {
			elem->weights[cnt].w = (struct element_t **)
			  obstack_alloc (&collate->mempool,
					 sizeof (struct element_t *));
			elem->weights[cnt].w[0] = elem;
			elem->weights[cnt].cnt = 1;
		      }
		    else
		      {
			/* Simply use the weight from `ellipsis_weight'.  */
			elem->weights[cnt].w =
			  collate->ellipsis_weight.weights[cnt].w;
			elem->weights[cnt].cnt =
			  collate->ellipsis_weight.weights[cnt].cnt;
		      }
		}

	      /* Increment for the next round.  */
	    increment:
	      for (cnt = len - 1; cnt >= 0; --cnt)
		if (++mbcnt[cnt] != '\0')
		  break;

	      /* Find out whether this was all.  */
	      if (cnt < 0 || memcmp (mbcnt, mbend, len) >= 0)
		/* Yep, that's all.  */
		break;
	    }
	}
    }
  else
    {
      /* For symbolic range we naturally must have a beginning and an
	 end specified by the user.  */
      if (startp == NULL)
	lr_error (ldfile, _("\
%s: symbolic range ellipsis must not directly follow `order_start'"),
		  "LC_COLLATE");
      else if (endp == NULL)
	lr_error (ldfile, _("\
%s: symbolic range ellipsis must not be directly followed by `order_end'"),
		  "LC_COLLATE");
      else
	{
	  /* Determine the range.  To do so we have to determine the
	     common prefix of the both names and then the numeric
	     values of both ends.  */
	  size_t lenfrom = strlen (startp->name);
	  size_t lento = strlen (endp->name);
	  char buf[lento + 1];
	  int preflen = 0;
	  long int from;
	  long int to;
	  char *cp;
	  int base = ellipsis == tok_ellipsis2 ? 16 : 10;

	  if (lenfrom != lento)
	    {
	    invalid_range:
	      lr_error (ldfile, _("\
`%s' and `%.*s' are not valid names for symbolic range"),
			startp->name, (int) lento, endp->name);
	      return;
	    }

	  while (startp->name[preflen] == endp->name[preflen])
	    if (startp->name[preflen] == '\0')
	      /* Nothing to be done.  The start and end point are identical
		 and while inserting the end point we have already given
		 the user an error message.  */
	      return;
	    else
	      ++preflen;

	  errno = 0;
	  from = strtol (startp->name + preflen, &cp, base);
	  if ((from == UINT_MAX && errno == ERANGE) || *cp != '\0')
	    goto invalid_range;

	  errno = 0;
	  to = strtol (endp->name + preflen, &cp, base);
	  if ((to == UINT_MAX && errno == ERANGE) || *cp != '\0')
	    goto invalid_range;

	  /* Copy the prefix.  */
	  memcpy (buf, startp->name, preflen);

	  /* Loop over all values.  */
	  for (++from; from < to; ++from)
	    {
	      struct element_t *elem = NULL;
	      struct charseq *seq;
	      uint32_t wc;
	      int cnt;

	      /* Generate the name.  */
	      sprintf (buf + preflen, base == 10 ? "%0*ld" : "%0*lX",
		       (int) (lenfrom - preflen), from);

	      /* Look whether this name is already defined.  */
	      void *ptr;
	      if (find_entry (&collate->seq_table, buf, symlen, &ptr) == 0)
		{
		  /* Copy back the result.  */
		  elem = ptr;

		  if (elem->next != NULL || (collate->cursor != NULL
					     && elem->next == collate->cursor))
		    {
		      lr_error (ldfile, _("\
%s: order for `%.*s' already defined at %s:%zu"),
				"LC_COLLATE", (int) lenfrom, buf,
				elem->file, elem->line);
		      continue;
		    }

		  if (elem->name == NULL)
		    {
		      lr_error (ldfile, _("%s: `%s' must be a character"),
				"LC_COLLATE", buf);
		      continue;
		    }
		}

	      if (elem == NULL || (elem->mbs == NULL && elem->wcs == NULL))
		{
		  /* Search for a character of this name.  */
		  seq = charmap_find_value (charmap, buf, lenfrom);
		  if (seq == NULL || seq->ucs4 == UNINITIALIZED_CHAR_VALUE)
		    {
		      wc = repertoire_find_value (repertoire, buf, lenfrom);

		      if (seq != NULL)
			seq->ucs4 = wc;
		    }
		  else
		    wc = seq->ucs4;

		  if (wc == ILLEGAL_CHAR_VALUE && seq == NULL)
		    /* We don't know anything about a character with this
		       name.  XXX Should we warn?  */
		    continue;

		  if (elem == NULL)
		    {
		      uint32_t wcs[2] = { wc, 0 };

		      /* We have to allocate an entry.  */
		      elem = new_element (collate,
					  seq != NULL
					  ? (char *) seq->bytes : NULL,
					  seq != NULL ? seq->nbytes : 0,
					  wc == ILLEGAL_CHAR_VALUE
					  ? NULL : wcs, buf, lenfrom, 1);
		    }
		  else
		    {
		      /* Update the element.  */
		      if (seq != NULL)
			{
			  elem->mbs = obstack_copy0 (&collate->mempool,
						     seq->bytes, seq->nbytes);
			  elem->nmbs = seq->nbytes;
			}

		      if (wc != ILLEGAL_CHAR_VALUE)
			{
			  uint32_t zero = 0;

			  obstack_grow (&collate->mempool,
					&wc, sizeof (uint32_t));
			  obstack_grow (&collate->mempool,
					&zero, sizeof (uint32_t));
			  elem->wcs = obstack_finish (&collate->mempool);
			  elem->nwcs = 1;
			}
		    }

		  elem->file = ldfile->fname;
		  elem->line = ldfile->lineno;
		  elem->section = collate->current_section;
		}

	      /* Enqueue the new element.  */
	      elem->last = collate->cursor;
	      elem->next = collate->cursor->next;
	      elem->last->next = elem;
	      if (elem->next != NULL)
		elem->next->last = elem;
	      collate->cursor = elem;

	      /* Now add the weights.  They come from the `ellipsis_weights'
		 member of `collate'.  */
	      elem->weights = (struct element_list_t *)
		obstack_alloc (&collate->mempool,
			       nrules * sizeof (struct element_list_t));
	      for (cnt = 0; cnt < nrules; ++cnt)
		if (collate->ellipsis_weight.weights[cnt].cnt == 1
		    && (collate->ellipsis_weight.weights[cnt].w[0]
			== ELEMENT_ELLIPSIS2))
		  {
		    elem->weights[cnt].w = (struct element_t **)
		      obstack_alloc (&collate->mempool,
				     sizeof (struct element_t *));
		    elem->weights[cnt].w[0] = elem;
		    elem->weights[cnt].cnt = 1;
		  }
		else
		  {
		    /* Simly use the weight from `ellipsis_weight'.  */
		    elem->weights[cnt].w =
		      collate->ellipsis_weight.weights[cnt].w;
		    elem->weights[cnt].cnt =
		      collate->ellipsis_weight.weights[cnt].cnt;
		  }
	    }
	}
    }
  /* Move the cursor to the last entry in the ellipsis.
     Subsequent operations need to start from the last entry.  */
  collate->cursor = endp;
}


static void
collate_startup (struct linereader *ldfile, struct localedef_t *locale,
		 struct localedef_t *copy_locale, int ignore_content)
{
  if (!ignore_content && locale->categories[LC_COLLATE].collate == NULL)
    {
      struct locale_collate_t *collate;

      if (copy_locale == NULL)
	{
	  collate = locale->categories[LC_COLLATE].collate =
	    (struct locale_collate_t *)
	    xcalloc (1, sizeof (struct locale_collate_t));

	  /* Init the various data structures.  */
	  init_hash (&collate->elem_table, 100);
	  init_hash (&collate->sym_table, 100);
	  init_hash (&collate->seq_table, 500);
	  obstack_init (&collate->mempool);

	  collate->col_weight_max = -1;
	}
      else
	/* Reuse the copy_locale's data structures.  */
	collate = locale->categories[LC_COLLATE].collate =
	  copy_locale->categories[LC_COLLATE].collate;
    }

  ldfile->translate_strings = 0;
  ldfile->return_widestr = 0;
}


void
collate_finish (struct localedef_t *locale, const struct charmap_t *charmap)
{
  /* Now is the time when we can assign the individual collation
     values for all the symbols.  We have possibly different values
     for the wide- and the multibyte-character symbols.  This is done
     since it might make a difference in the encoding if there is in
     some cases no multibyte-character but there are wide-characters.
     (The other way around it is not important since theencoded
     collation value in the wide-character case is 32 bits wide and
     therefore requires no encoding).

     The lowest collation value assigned is 2.  Zero is reserved for
     the NUL byte terminating the strings in the `strxfrm'/`wcsxfrm'
     functions and 1 is used to separate the individual passes for the
     different rules.

     We also have to construct is list with all the bytes/words which
     can come first in a sequence, followed by all the elements which
     also start with this byte/word.  The order is reverse which has
     among others the important effect that longer strings are located
     first in the list.  This is required for the output data since
     the algorithm used in `strcoll' etc depends on this.

     The multibyte case is easy.  We simply sort into an array with
     256 elements.  */
  struct locale_collate_t *collate = locale->categories[LC_COLLATE].collate;
  int mbact[nrules];
  int wcact;
  int mbseqact;
  int wcseqact;
  struct element_t *runp;
  int i;
  int need_undefined = 0;
  struct section_list *sect;
  int ruleidx;
  int nr_wide_elems = 0;

  if (collate == NULL)
    {
      /* No data, no check. Issue a warning.  */
      record_warning (_("No definition for %s category found"),
		      "LC_COLLATE");
      return;
    }

  /* If this assertion is hit change the type in `element_t'.  */
  assert (nrules <= sizeof (runp->used_in_level) * 8);

  /* Make sure that the `position' rule is used either in all sections
     or in none.  */
  for (i = 0; i < nrules; ++i)
    for (sect = collate->sections; sect != NULL; sect = sect->next)
      if (sect != collate->current_section
	  && sect->rules != NULL
	  && ((sect->rules[i] & sort_position)
	      != (collate->current_section->rules[i] & sort_position)))
	{
	  record_error (0, 0, _("\
%s: `position' must be used for a specific level in all sections or none"),
			"LC_COLLATE");
	  break;
	}

  /* Find out which elements are used at which level.  At the same
     time we find out whether we have any undefined symbols.  */
  runp = collate->start;
  while (runp != NULL)
    {
      if (runp->mbs != NULL)
	{
	  for (i = 0; i < nrules; ++i)
	    {
	      int j;

	      for (j = 0; j < runp->weights[i].cnt; ++j)
		/* A NULL pointer as the weight means IGNORE.  */
		if (runp->weights[i].w[j] != NULL)
		  {
		    if (runp->weights[i].w[j]->weights == NULL)
		      {
			record_error_at_line (0, 0, runp->file, runp->line,
					      _("symbol `%s' not defined"),
					      runp->weights[i].w[j]->name);

			need_undefined = 1;
			runp->weights[i].w[j] = &collate->undefined;
		      }
		    else
		      /* Set the bit for the level.  */
		      runp->weights[i].w[j]->used_in_level |= 1 << i;
		  }
	    }
	}

      /* Up to the next entry.  */
      runp = runp->next;
    }

  /* Walk through the list of defined sequences and assign weights.  Also
     create the data structure which will allow generating the single byte
     character based tables.

     Since at each time only the weights for each of the rules are
     only compared to other weights for this rule it is possible to
     assign more compact weight values than simply counting all
     weights in sequence.  We can assign weights from 3, one for each
     rule individually and only for those elements, which are actually
     used for this rule.

     Why is this important?  It is not for the wide char table.  But
     it is for the singlebyte output since here larger numbers have to
     be encoded to make it possible to emit the value as a byte
     string.  */
  for (i = 0; i < nrules; ++i)
    mbact[i] = 2;
  wcact = 2;
  mbseqact = 0;
  wcseqact = 0;
  runp = collate->start;
  while (runp != NULL)
    {
      /* Determine the order.  */
      if (runp->used_in_level != 0)
	{
	  runp->mborder = (int *) obstack_alloc (&collate->mempool,
						 nrules * sizeof (int));

	  for (i = 0; i < nrules; ++i)
	    if ((runp->used_in_level & (1 << i)) != 0)
	      runp->mborder[i] = mbact[i]++;
	    else
	      runp->mborder[i] = 0;
	}

      if (runp->mbs != NULL)
	{
	  struct element_t **eptr;
	  struct element_t *lastp = NULL;

	  /* Find the point where to insert in the list.  */
	  eptr = &collate->mbheads[((unsigned char *) runp->mbs)[0]];
	  while (*eptr != NULL)
	    {
	      if ((*eptr)->nmbs < runp->nmbs)
		break;

	      if ((*eptr)->nmbs == runp->nmbs)
		{
		  int c = memcmp ((*eptr)->mbs, runp->mbs, runp->nmbs);

		  if (c == 0)
		    {
		      /* This should not happen.  It means that we have
			 to symbols with the same byte sequence.  It is
			 of course an error.  */
		      record_error_at_line (0, 0, (*eptr)->file,
					    (*eptr)->line,
					    _("\
symbol `%s' has the same encoding as"), (*eptr)->name);

		      record_error_at_line (0, 0, runp->file, runp->line,
					    _("symbol `%s'"), runp->name);
		      goto dont_insert;
		    }
		  else if (c < 0)
		    /* Insert it here.  */
		    break;
		}

	      /* To the next entry.  */
	      lastp = *eptr;
	      eptr = &(*eptr)->mbnext;
	    }

	  /* Set the pointers.  */
	  runp->mbnext = *eptr;
	  runp->mblast = lastp;
	  if (*eptr != NULL)
	    (*eptr)->mblast = runp;
	  *eptr = runp;
	dont_insert:
	  ;
	}

      if (runp->used_in_level)
	{
	  runp->wcorder = wcact++;

	  /* We take the opportunity to count the elements which have
	     wide characters.  */
	  ++nr_wide_elems;
	}

      if (runp->is_character)
	{
	  if (runp->nmbs == 1)
	    collate->mbseqorder[((unsigned char *) runp->mbs)[0]] = mbseqact++;

	  runp->wcseqorder = wcseqact++;
	}
      else if (runp->mbs != NULL && runp->weights != NULL)
	/* This is for collation elements.  */
	runp->wcseqorder = wcseqact++;

      /* Up to the next entry.  */
      runp = runp->next;
    }

  /* Find out whether any of the `mbheads' entries is unset.  In this
     case we use the UNDEFINED entry.  */
  for (i = 1; i < 256; ++i)
    if (collate->mbheads[i] == NULL)
      {
	need_undefined = 1;
	collate->mbheads[i] = &collate->undefined;
      }

  /* Now to the wide character case.  */
  collate->wcheads.p = 6;
  collate->wcheads.q = 10;
  wchead_table_init (&collate->wcheads);

  collate->wcseqorder.p = 6;
  collate->wcseqorder.q = 10;
  collseq_table_init (&collate->wcseqorder);

  /* Start adding.  */
  runp = collate->start;
  while (runp != NULL)
    {
      if (runp->wcs != NULL)
	{
	  struct element_t *e;
	  struct element_t **eptr;
	  struct element_t *lastp;

	  /* Insert the collation sequence value.  */
	  if (runp->is_character)
	    collseq_table_add (&collate->wcseqorder, runp->wcs[0],
			       runp->wcseqorder);

	  /* Find the point where to insert in the list.  */
	  e = wchead_table_get (&collate->wcheads, runp->wcs[0]);
	  eptr = &e;
	  lastp = NULL;
	  while (*eptr != NULL)
	    {
	      if ((*eptr)->nwcs < runp->nwcs)
		break;

	      if ((*eptr)->nwcs == runp->nwcs)
		{
		  int c = wmemcmp ((wchar_t *) (*eptr)->wcs,
				   (wchar_t *) runp->wcs, runp->nwcs);

		  if (c == 0)
		    {
		      /* This should not happen.  It means that we have
			 two symbols with the same byte sequence.  It is
			 of course an error.  */
		      record_error_at_line (0, 0, (*eptr)->file,
					    (*eptr)->line,
					    _("\
symbol `%s' has the same encoding as"), (*eptr)->name);

		      record_error_at_line (0, 0, runp->file, runp->line,
					    _("symbol `%s'"), runp->name);
		      goto dont_insertwc;
		    }
		  else if (c < 0)
		    /* Insert it here.  */
		    break;
		}

	      /* To the next entry.  */
	      lastp = *eptr;
	      eptr = &(*eptr)->wcnext;
	    }

	  /* Set the pointers.  */
	  runp->wcnext = *eptr;
	  runp->wclast = lastp;
	  if (*eptr != NULL)
	    (*eptr)->wclast = runp;
	  *eptr = runp;
	  if (eptr == &e)
	    wchead_table_add (&collate->wcheads, runp->wcs[0], e);
	dont_insertwc:
	  ;
	}

      /* Up to the next entry.  */
      runp = runp->next;
    }

  /* Now determine whether the UNDEFINED entry is needed and if yes,
     whether it was defined.  */
  collate->undefined.used_in_level = need_undefined ? ~0u : 0;
  if (collate->undefined.file == NULL)
    {
      if (need_undefined)
	{
	  /* This seems not to be enforced by recent standards.  Don't
	     emit an error, simply append UNDEFINED at the end.  */
	  collate->undefined.mborder =
	    (int *) obstack_alloc (&collate->mempool, nrules * sizeof (int));

	  for (i = 0; i < nrules; ++i)
	    collate->undefined.mborder[i] = mbact[i]++;
	}

      /* In any case we will need the definition for the wide character
	 case.  But we will not complain that it is missing since the
	 specification strangely enough does not seem to account for
	 this.  */
      collate->undefined.wcorder = wcact++;
    }

  /* Finally, try to unify the rules for the sections.  Whenever the rules
     for a section are the same as those for another section give the
     ruleset the same index.  Since there are never many section we can
     use an O(n^2) algorithm here.  */
  sect = collate->sections;
  while (sect != NULL && sect->rules == NULL)
    sect = sect->next;

  /* Bail out if we have no sections because of earlier errors.  */
  if (sect == NULL)
    {
      record_error (EXIT_FAILURE, 0, _("too many errors; giving up"));
      return;
    }

  ruleidx = 0;
  do
    {
      struct section_list *osect = collate->sections;

      while (osect != sect)
	if (osect->rules != NULL
	    && memcmp (osect->rules, sect->rules,
		       nrules * sizeof (osect->rules[0])) == 0)
	  break;
	else
	  osect = osect->next;

      if (osect == sect)
	sect->ruleidx = ruleidx++;
      else
	sect->ruleidx = osect->ruleidx;

      /* Next section.  */
      do
	sect = sect->next;
      while (sect != NULL && sect->rules == NULL);
    }
  while (sect != NULL);
  /* We are currently not prepared for more than 128 rulesets.  But this
     should never really be a problem.  */
  assert (ruleidx <= 128);
}


static int32_t
output_weight (struct obstack *pool, struct locale_collate_t *collate,
	       struct element_t *elem)
{
  size_t cnt;
  int32_t retval;

  /* Optimize the use of UNDEFINED.  */
  if (elem == &collate->undefined)
    /* The weights are already inserted.  */
    return 0;

  /* This byte can start exactly one collation element and this is
     a single byte.  We can directly give the index to the weights.  */
  retval = obstack_object_size (pool);

  /* Construct the weight.  */
  for (cnt = 0; cnt < nrules; ++cnt)
    {
      char buf[elem->weights[cnt].cnt * 7];
      int len = 0;
      int i;

      for (i = 0; i < elem->weights[cnt].cnt; ++i)
	/* Encode the weight value.  We do nothing for IGNORE entries.  */
	if (elem->weights[cnt].w[i] != NULL)
	  len += utf8_encode (&buf[len],
			      elem->weights[cnt].w[i]->mborder[cnt]);

      /* And add the buffer content.  */
      obstack_1grow (pool, len);
      obstack_grow (pool, buf, len);
    }

  return retval | ((elem->section->ruleidx & 0x7f) << 24);
}


static int32_t
output_weightwc (struct obstack *pool, struct locale_collate_t *collate,
		 struct element_t *elem)
{
  size_t cnt;
  int32_t retval;

  /* Optimize the use of UNDEFINED.  */
  if (elem == &collate->undefined)
    /* The weights are already inserted.  */
    return 0;

  /* This byte can start exactly one collation element and this is
     a single byte.  We can directly give the index to the weights.  */
  retval = obstack_object_size (pool) / sizeof (int32_t);

  /* Construct the weight.  */
  for (cnt = 0; cnt < nrules; ++cnt)
    {
      int32_t buf[elem->weights[cnt].cnt];
      int i;
      int32_t j;

      for (i = 0, j = 0; i < elem->weights[cnt].cnt; ++i)
	if (elem->weights[cnt].w[i] != NULL)
	  buf[j++] = elem->weights[cnt].w[i]->wcorder;

      /* And add the buffer content.  */
      obstack_int32_grow (pool, j);

      obstack_grow (pool, buf, j * sizeof (int32_t));
      maybe_swap_uint32_obstack (pool, j);
    }

  return retval | ((elem->section->ruleidx & 0x7f) << 24);
}

/* If localedef is every threaded, this would need to be __thread var.  */
static struct
{
  struct obstack *weightpool;
  struct obstack *extrapool;
  struct obstack *indpool;
  struct locale_collate_t *collate;
  struct collidx_table *tablewc;
} atwc;

static void add_to_tablewc (uint32_t ch, struct element_t *runp);

static void
add_to_tablewc (uint32_t ch, struct element_t *runp)
{
  if (runp->wcnext == NULL && runp->nwcs == 1)
    {
      int32_t weigthidx = output_weightwc (atwc.weightpool, atwc.collate,
					   runp);
      collidx_table_add (atwc.tablewc, ch, weigthidx);
    }
  else
    {
      /* As for the singlebyte table, we recognize sequences and
	 compress them.  */

      collidx_table_add (atwc.tablewc, ch,
			 -(obstack_object_size (atwc.extrapool)
			 / sizeof (uint32_t)));

      do
	{
	  /* Store the current index in the weight table.  We know that
	     the current position in the `extrapool' is aligned on a
	     32-bit address.  */
	  int32_t weightidx;
	  int added;

	  /* Find out wether this is a single entry or we have more than
	     one consecutive entry.  */
	  if (runp->wcnext != NULL
	      && runp->nwcs == runp->wcnext->nwcs
	      && wmemcmp ((wchar_t *) runp->wcs,
			  (wchar_t *)runp->wcnext->wcs,
			  runp->nwcs - 1) == 0
	      && (runp->wcs[runp->nwcs - 1]
		  == runp->wcnext->wcs[runp->nwcs - 1] + 1))
	    {
	      int i;
	      struct element_t *series_startp = runp;
	      struct element_t *curp;

	      /* Now add first the initial byte sequence.  */
	      added = (1 + 1 + 2 * (runp->nwcs - 1)) * sizeof (int32_t);
	      if (sizeof (int32_t) == sizeof (int))
		obstack_make_room (atwc.extrapool, added);

	      /* More than one consecutive entry.  We mark this by having
		 a negative index into the indirect table.  */
	      obstack_int32_grow_fast (atwc.extrapool,
				       -(obstack_object_size (atwc.indpool)
					 / sizeof (int32_t)));
	      obstack_int32_grow_fast (atwc.extrapool, runp->nwcs - 1);

	      do
		runp = runp->wcnext;
	      while (runp->wcnext != NULL
		     && runp->nwcs == runp->wcnext->nwcs
		     && wmemcmp ((wchar_t *) runp->wcs,
				 (wchar_t *)runp->wcnext->wcs,
				 runp->nwcs - 1) == 0
		     && (runp->wcs[runp->nwcs - 1]
			 == runp->wcnext->wcs[runp->nwcs - 1] + 1));

	      /* Now walk backward from here to the beginning.  */
	      curp = runp;

	      for (i = 1; i < runp->nwcs; ++i)
		obstack_int32_grow_fast (atwc.extrapool, curp->wcs[i]);

	      /* Now find the end of the consecutive sequence and
		 add all the indices in the indirect pool.  */
	      do
		{
		  weightidx = output_weightwc (atwc.weightpool, atwc.collate,
					       curp);
		  obstack_int32_grow (atwc.indpool, weightidx);

		  curp = curp->wclast;
		}
	      while (curp != series_startp);

	      /* Add the final weight.  */
	      weightidx = output_weightwc (atwc.weightpool, atwc.collate,
					   curp);
	      obstack_int32_grow (atwc.indpool, weightidx);

	      /* And add the end byte sequence.  Without length this
		 time.  */
	      for (i = 1; i < curp->nwcs; ++i)
		obstack_int32_grow (atwc.extrapool, curp->wcs[i]);
	    }
	  else
	    {
	      /* A single entry.  Simply add the index and the length and
		 string (except for the first character which is already
		 tested for).  */
	      int i;

	      /* Output the weight info.  */
	      weightidx = output_weightwc (atwc.weightpool, atwc.collate,
					   runp);

	      assert (runp->nwcs > 0);
	      added = (1 + 1 + runp->nwcs - 1) * sizeof (int32_t);
	      if (sizeof (int) == sizeof (int32_t))
		obstack_make_room (atwc.extrapool, added);

	      obstack_int32_grow_fast (atwc.extrapool, weightidx);
	      obstack_int32_grow_fast (atwc.extrapool, runp->nwcs - 1);
	      for (i = 1; i < runp->nwcs; ++i)
		obstack_int32_grow_fast (atwc.extrapool, runp->wcs[i]);
	    }

	  /* Next entry.  */
	  runp = runp->wcnext;
	}
      while (runp != NULL);
    }
}

void
collate_output (struct localedef_t *locale, const struct charmap_t *charmap,
		const char *output_path)
{
  struct locale_collate_t *collate = locale->categories[LC_COLLATE].collate;
  const size_t nelems = _NL_ITEM_INDEX (_NL_NUM_LC_COLLATE);
  struct locale_file file;
  size_t ch;
  int32_t tablemb[256];
  struct obstack weightpool;
  struct obstack extrapool;
  struct obstack indirectpool;
  struct section_list *sect;
  struct collidx_table tablewc;
  uint32_t elem_size;
  uint32_t *elem_table;
  int i;
  struct element_t *runp;

  init_locale_data (&file, nelems);
  add_locale_uint32 (&file, nrules);

  /* If we have no LC_COLLATE data emit only the number of rules as zero.  */
  if (collate == NULL)
    {
      size_t idx;
      for (idx = 1; idx < nelems; idx++)
	{
	  /* The words have to be handled specially.  */
	  if (idx == _NL_ITEM_INDEX (_NL_COLLATE_SYMB_HASH_SIZEMB))
	    add_locale_uint32 (&file, 0);
	  else
	    add_locale_empty (&file);
	}
      write_locale_data (output_path, LC_COLLATE, "LC_COLLATE", &file);
      return;
    }

  obstack_init (&weightpool);
  obstack_init (&extrapool);
  obstack_init (&indirectpool);

  /* Since we are using the sign of an integer to mark indirection the
     offsets in the arrays we are indirectly referring to must not be
     zero since -0 == 0.  Therefore we add a bit of dummy content.  */
  obstack_int32_grow (&extrapool, 0);
  obstack_int32_grow (&indirectpool, 0);

  /* Prepare the ruleset table.  */
  for (sect = collate->sections, i = 0; sect != NULL; sect = sect->next)
    if (sect->rules != NULL && sect->ruleidx == i)
      {
	int j;

	obstack_make_room (&weightpool, nrules);

	for (j = 0; j < nrules; ++j)
	  obstack_1grow_fast (&weightpool, sect->rules[j]);
	++i;
      }
  /* And align the output.  */
  i = (nrules * i) % LOCFILE_ALIGN;
  if (i > 0)
    do
      obstack_1grow (&weightpool, '\0');
    while (++i < LOCFILE_ALIGN);

  add_locale_raw_obstack (&file, &weightpool);

  /* Generate the 8-bit table.  Walk through the lists of sequences
     starting with the same byte and add them one after the other to
     the table.  In case we have more than one sequence starting with
     the same byte we have to use extra indirection.

     First add a record for the NUL byte.  This entry will never be used
     so it does not matter.  */
  tablemb[0] = 0;

  /* Now insert the `UNDEFINED' value if it is used.  Since this value
     will probably be used more than once it is good to store the
     weights only once.  */
  if (collate->undefined.used_in_level != 0)
    output_weight (&weightpool, collate, &collate->undefined);

  for (ch = 1; ch < 256; ++ch)
    if (collate->mbheads[ch]->mbnext == NULL
	&& collate->mbheads[ch]->nmbs <= 1)
      {
	tablemb[ch] = output_weight (&weightpool, collate,
				     collate->mbheads[ch]);
      }
    else
      {
	/* The entries in the list are sorted by length and then
	   alphabetically.  This is the order in which we will add the
	   elements to the collation table.  This allows simply walking
	   the table in sequence and stopping at the first matching
	   entry.  Since the longer sequences are coming first in the
	   list they have the possibility to match first, just as it
	   has to be.  In the worst case we are walking to the end of
	   the list where we put, if no singlebyte sequence is defined
	   in the locale definition, the weights for UNDEFINED.

	   To reduce the length of the search list we compress them a bit.
	   This happens by collecting sequences of consecutive byte
	   sequences in one entry (having and begin and end byte sequence)
	   and add only one index into the weight table.  We can find the
	   consecutive entries since they are also consecutive in the list.  */
	struct element_t *runp = collate->mbheads[ch];
	struct element_t *lastp;

	assert (LOCFILE_ALIGNED_P (obstack_object_size (&extrapool)));

	tablemb[ch] = -obstack_object_size (&extrapool);

	do
	  {
	    /* Store the current index in the weight table.  We know that
	       the current position in the `extrapool' is aligned on a
	       32-bit address.  */
	    int32_t weightidx;
	    int added;

	    /* Find out wether this is a single entry or we have more than
	       one consecutive entry.  */
	    if (runp->mbnext != NULL
		&& runp->nmbs == runp->mbnext->nmbs
		&& memcmp (runp->mbs, runp->mbnext->mbs, runp->nmbs - 1) == 0
		&& (runp->mbs[runp->nmbs - 1]
		    == runp->mbnext->mbs[runp->nmbs - 1] + 1))
	      {
		int i;
		struct element_t *series_startp = runp;
		struct element_t *curp;

		/* Compute how much space we will need.  */
		added = LOCFILE_ALIGN_UP (sizeof (int32_t) + 1
					  + 2 * (runp->nmbs - 1));
		assert (LOCFILE_ALIGNED_P (obstack_object_size (&extrapool)));
		obstack_make_room (&extrapool, added);

		/* More than one consecutive entry.  We mark this by having
		   a negative index into the indirect table.  */
		obstack_int32_grow_fast (&extrapool,
					 -(obstack_object_size (&indirectpool)
					   / sizeof (int32_t)));

		/* Now search first the end of the series.  */
		do
		  runp = runp->mbnext;
		while (runp->mbnext != NULL
		       && runp->nmbs == runp->mbnext->nmbs
		       && memcmp (runp->mbs, runp->mbnext->mbs,
				  runp->nmbs - 1) == 0
		       && (runp->mbs[runp->nmbs - 1]
			   == runp->mbnext->mbs[runp->nmbs - 1] + 1));

		/* Now walk backward from here to the beginning.  */
		curp = runp;

		assert (runp->nmbs <= 256);
		obstack_1grow_fast (&extrapool, curp->nmbs - 1);
		for (i = 1; i < curp->nmbs; ++i)
		  obstack_1grow_fast (&extrapool, curp->mbs[i]);

		/* Now find the end of the consecutive sequence and
		   add all the indices in the indirect pool.  */
		do
		  {
		    weightidx = output_weight (&weightpool, collate, curp);
		    obstack_int32_grow (&indirectpool, weightidx);

		    curp = curp->mblast;
		  }
		while (curp != series_startp);

		/* Add the final weight.  */
		weightidx = output_weight (&weightpool, collate, curp);
		obstack_int32_grow (&indirectpool, weightidx);

		/* And add the end byte sequence.  Without length this
		   time.  */
		for (i = 1; i < curp->nmbs; ++i)
		  obstack_1grow_fast (&extrapool, curp->mbs[i]);
	      }
	    else
	      {
		/* A single entry.  Simply add the index and the length and
		   string (except for the first character which is already
		   tested for).  */
		int i;

		/* Output the weight info.  */
		weightidx = output_weight (&weightpool, collate, runp);

		added = LOCFILE_ALIGN_UP (sizeof (int32_t) + 1
					  + runp->nmbs - 1);
		assert (LOCFILE_ALIGNED_P (obstack_object_size (&extrapool)));
		obstack_make_room (&extrapool, added);

		obstack_int32_grow_fast (&extrapool, weightidx);
		assert (runp->nmbs <= 256);
		obstack_1grow_fast (&extrapool, runp->nmbs - 1);

		for (i = 1; i < runp->nmbs; ++i)
		  obstack_1grow_fast (&extrapool, runp->mbs[i]);
	      }

	    /* Add alignment bytes if necessary.  */
	    while (!LOCFILE_ALIGNED_P (obstack_object_size (&extrapool)))
	      obstack_1grow_fast (&extrapool, '\0');

	    /* Next entry.  */
	    lastp = runp;
	    runp = runp->mbnext;
	  }
	while (runp != NULL);

	assert (LOCFILE_ALIGNED_P (obstack_object_size (&extrapool)));

	/* If the final entry in the list is not a single character we
	   add an UNDEFINED entry here.  */
	if (lastp->nmbs != 1)
	  {
	    int added = LOCFILE_ALIGN_UP (sizeof (int32_t) + 1 + 1);
	    obstack_make_room (&extrapool, added);

	    obstack_int32_grow_fast (&extrapool, 0);
	    /* XXX What rule? We just pick the first.  */
	    obstack_1grow_fast (&extrapool, 0);
	    /* Length is zero.  */
	    obstack_1grow_fast (&extrapool, 0);

	    /* Add alignment bytes if necessary.  */
	    while (!LOCFILE_ALIGNED_P (obstack_object_size (&extrapool)))
	      obstack_1grow_fast (&extrapool, '\0');
	  }
      }

  /* Add padding to the tables if necessary.  */
  while (!LOCFILE_ALIGNED_P (obstack_object_size (&weightpool)))
    obstack_1grow (&weightpool, 0);

  /* Now add the four tables.  */
  add_locale_uint32_array (&file, (const uint32_t *) tablemb, 256);
  add_locale_raw_obstack (&file, &weightpool);
  add_locale_raw_obstack (&file, &extrapool);
  add_locale_raw_obstack (&file, &indirectpool);

  /* Now the same for the wide character table.  We need to store some
     more information here.  */
  add_locale_empty (&file);
  add_locale_empty (&file);
  add_locale_empty (&file);

  /* Since we are using the sign of an integer to mark indirection the
     offsets in the arrays we are indirectly referring to must not be
     zero since -0 == 0.  Therefore we add a bit of dummy content.  */
  obstack_int32_grow (&extrapool, 0);
  obstack_int32_grow (&indirectpool, 0);

  /* Now insert the `UNDEFINED' value if it is used.  Since this value
     will probably be used more than once it is good to store the
     weights only once.  */
  if (output_weightwc (&weightpool, collate, &collate->undefined) != 0)
    abort ();

  /* Generate the table.  Walk through the lists of sequences starting
     with the same wide character and add them one after the other to
     the table.  In case we have more than one sequence starting with
     the same byte we have to use extra indirection.  */
  tablewc.p = 6;
  tablewc.q = 10;
  collidx_table_init (&tablewc);

  atwc.weightpool = &weightpool;
  atwc.extrapool = &extrapool;
  atwc.indpool = &indirectpool;
  atwc.collate = collate;
  atwc.tablewc = &tablewc;

  wchead_table_iterate (&collate->wcheads, add_to_tablewc);

  memset (&atwc, 0, sizeof (atwc));

  /* Now add the four tables.  */
  add_locale_collidx_table (&file, &tablewc);
  add_locale_raw_obstack (&file, &weightpool);
  add_locale_raw_obstack (&file, &extrapool);
  add_locale_raw_obstack (&file, &indirectpool);

  /* Finally write the table with collation element names out.  It is
     a hash table with a simple function which gets the name of the
     character as the input.  One character might have many names.  The
     value associated with the name is an index into the weight table
     where we are then interested in the first-level weight value.

     To determine how large the table should be we are counting the
     elements have to put in.  Since we are using internal chaining
     using a secondary hash function we have to make the table a bit
     larger to avoid extremely long search times.  We can achieve
     good results with a 40% larger table than there are entries.  */
  elem_size = 0;
  runp = collate->start;
  while (runp != NULL)
    {
      if (runp->mbs != NULL && runp->weights != NULL && !runp->is_character)
	/* Yep, the element really counts.  */
	++elem_size;

      runp = runp->next;
    }
  /* Add 50% and find the next prime number.  */
  elem_size = next_prime (elem_size + (elem_size >> 1));

  /* Allocate the table.  Each entry consists of two words: the hash
     value and an index in a secondary table which provides the index
     into the weight table and the string itself (so that a match can
     be determined).  */
  elem_table = (uint32_t *) obstack_alloc (&extrapool,
					   elem_size * 2 * sizeof (uint32_t));
  memset (elem_table, '\0', elem_size * 2 * sizeof (uint32_t));

  /* Now add the elements.  */
  runp = collate->start;
  while (runp != NULL)
    {
      if (runp->mbs != NULL && runp->weights != NULL && !runp->is_character)
	{
	  /* Compute the hash value of the name.  */
	  uint32_t namelen = strlen (runp->name);
	  uint32_t hash = elem_hash (runp->name, namelen);
	  size_t idx = hash % elem_size;
#ifndef NDEBUG
	  size_t start_idx = idx;
#endif

	  if (elem_table[idx * 2] != 0)
	    {
	      /* The spot is already taken.  Try iterating using the value
		 from the secondary hashing function.  */
	      size_t iter = hash % (elem_size - 2) + 1;

	      do
		{
		  idx += iter;
		  if (idx >= elem_size)
		    idx -= elem_size;
		  assert (idx != start_idx);
		}
	      while (elem_table[idx * 2] != 0);
	    }
	  /* This is the spot where we will insert the value.  */
 	  elem_table[idx * 2] = hash;
	  elem_table[idx * 2 + 1] = obstack_object_size (&extrapool);

	  /* The string itself including length.  */
	  obstack_1grow (&extrapool, namelen);
	  obstack_grow (&extrapool, runp->name, namelen);

	  /* And the multibyte representation.  */
	  obstack_1grow (&extrapool, runp->nmbs);
	  obstack_grow (&extrapool, runp->mbs, runp->nmbs);

	  /* And align again to 32 bits.  */
	  if ((1 + namelen + 1 + runp->nmbs) % sizeof (int32_t) != 0)
	    obstack_grow (&extrapool, "\0\0",
			  (sizeof (int32_t)
			   - ((1 + namelen + 1 + runp->nmbs)
			      % sizeof (int32_t))));

	  /* Now some 32-bit values: multibyte collation sequence,
	     wide char string (including length), and wide char
	     collation sequence.  */
	  obstack_int32_grow (&extrapool, runp->mbseqorder);

	  obstack_int32_grow (&extrapool, runp->nwcs);
	  obstack_grow (&extrapool, runp->wcs,
			runp->nwcs * sizeof (uint32_t));
	  maybe_swap_uint32_obstack (&extrapool, runp->nwcs);

	  obstack_int32_grow (&extrapool, runp->wcseqorder);
	}

      runp = runp->next;
    }

  /* Prepare to write out this data.  */
  add_locale_uint32 (&file, elem_size);
  add_locale_uint32_array (&file, elem_table, 2 * elem_size);
  add_locale_raw_obstack (&file, &extrapool);
  add_locale_raw_data (&file, collate->mbseqorder, 256);
  add_locale_collseq_table (&file, &collate->wcseqorder);
  add_locale_string (&file, charmap->code_set_name);
  write_locale_data (output_path, LC_COLLATE, "LC_COLLATE", &file);

  obstack_free (&weightpool, NULL);
  obstack_free (&extrapool, NULL);
  obstack_free (&indirectpool, NULL);
}


static enum token_t
skip_to (struct linereader *ldfile, struct locale_collate_t *collate,
	 const struct charmap_t *charmap, int to_endif)
{
  while (1)
    {
      struct token *now = lr_token (ldfile, charmap, NULL, NULL, 0);
      enum token_t nowtok = now->tok;

      if (nowtok == tok_eof || nowtok == tok_end)
	return nowtok;

      if (nowtok == tok_ifdef || nowtok == tok_ifndef)
	{
	  lr_error (ldfile, _("%s: nested conditionals not supported"),
		    "LC_COLLATE");
	  nowtok = skip_to (ldfile, collate, charmap, tok_endif);
	  if (nowtok == tok_eof || nowtok == tok_end)
	    return nowtok;
	}
      else if (nowtok == tok_endif || (!to_endif && nowtok == tok_else))
	{
	  lr_ignore_rest (ldfile, 1);
	  return nowtok;
	}
      else if (!to_endif && (nowtok == tok_elifdef || nowtok == tok_elifndef))
	{
	  /* Do not read the rest of the line.  */
	  return nowtok;
	}
      else if (nowtok == tok_else)
	{
	  lr_error (ldfile, _("%s: more than one 'else'"), "LC_COLLATE");
	}

      lr_ignore_rest (ldfile, 0);
    }
}


void
collate_read (struct linereader *ldfile, struct localedef_t *result,
	      const struct charmap_t *charmap, const char *repertoire_name,
	      int ignore_content)
{
  struct repertoire_t *repertoire = NULL;
  struct locale_collate_t *collate;
  struct token *now;
  struct token *arg = NULL;
  enum token_t nowtok;
  enum token_t was_ellipsis = tok_none;
  struct localedef_t *copy_locale = NULL;
  /* Parsing state:
     0 - start
     1 - between `order-start' and `order-end'
     2 - after `order-end'
     3 - after `reorder-after', waiting for `reorder-end'
     4 - after `reorder-end'
     5 - after `reorder-sections-after', waiting for `reorder-sections-end'
     6 - after `reorder-sections-end'
  */
  int state = 0;

  /* Get the repertoire we have to use.  */
  if (repertoire_name != NULL)
    repertoire = repertoire_read (repertoire_name);

  /* The rest of the line containing `LC_COLLATE' must be free.  */
  lr_ignore_rest (ldfile, 1);

  while (1)
    {
      do
	{
	  now = lr_token (ldfile, charmap, result, NULL, verbose);
	  nowtok = now->tok;
	}
      while (nowtok == tok_eol);

      if (nowtok != tok_define)
	break;

      if (ignore_content)
	lr_ignore_rest (ldfile, 0);
      else
	{
	  arg = lr_token (ldfile, charmap, result, NULL, verbose);
	  if (arg->tok != tok_ident)
	    SYNTAX_ERROR (_("%s: syntax error"), "LC_COLLATE");
	  else
	    {
	      /* Simply add the new symbol.  */
	      struct name_list *newsym = xmalloc (sizeof (*newsym)
						  + arg->val.str.lenmb + 1);
	      memcpy (newsym->str, arg->val.str.startmb, arg->val.str.lenmb);
	      newsym->str[arg->val.str.lenmb] = '\0';
	      newsym->next = defined;
	      defined = newsym;

	      lr_ignore_rest (ldfile, 1);
	    }
	}
    }

  if (nowtok == tok_copy)
    {
      now = lr_token (ldfile, charmap, result, NULL, verbose);
      if (now->tok != tok_string)
	{
	  SYNTAX_ERROR (_("%s: syntax error"), "LC_COLLATE");

	skip_category:
	  do
	    now = lr_token (ldfile, charmap, result, NULL, verbose);
	  while (now->tok != tok_eof && now->tok != tok_end);

	  if (now->tok != tok_eof
	      || (now = lr_token (ldfile, charmap, result, NULL, verbose),
		  now->tok == tok_eof))
	    lr_error (ldfile, _("%s: premature end of file"), "LC_COLLATE");
	  else if (now->tok != tok_lc_collate)
	    {
	      lr_error (ldfile, _("\
%1$s: definition does not end with `END %1$s'"), "LC_COLLATE");
	      lr_ignore_rest (ldfile, 0);
	    }
	  else
	    lr_ignore_rest (ldfile, 1);

	  return;
	}

      if (! ignore_content)
	{
	  /* Get the locale definition.  */
	  copy_locale = load_locale (LC_COLLATE, now->val.str.startmb,
				     repertoire_name, charmap, NULL);
	  if ((copy_locale->avail & COLLATE_LOCALE) == 0)
	    {
	      /* Not yet loaded.  So do it now.  */
	      if (locfile_read (copy_locale, charmap) != 0)
		goto skip_category;
	    }

	  if (copy_locale->categories[LC_COLLATE].collate == NULL)
	    return;
	}

      lr_ignore_rest (ldfile, 1);

      now = lr_token (ldfile, charmap, result, NULL, verbose);
      nowtok = now->tok;
    }

  /* Prepare the data structures.  */
  collate_startup (ldfile, result, copy_locale, ignore_content);
  collate = result->categories[LC_COLLATE].collate;

  while (1)
    {
      char ucs4buf[10];
      char *symstr;
      size_t symlen;

      /* Of course we don't proceed beyond the end of file.  */
      if (nowtok == tok_eof)
	break;

      /* Ingore empty lines.  */
      if (nowtok == tok_eol)
	{
	  now = lr_token (ldfile, charmap, result, NULL, verbose);
	  nowtok = now->tok;
	  continue;
	}

      switch (nowtok)
	{
	case tok_copy:
	  /* Allow copying other locales.  */
	  now = lr_token (ldfile, charmap, result, NULL, verbose);
	  if (now->tok != tok_string)
	    goto err_label;

	  if (! ignore_content)
	    load_locale (LC_COLLATE, now->val.str.startmb, repertoire_name,
			 charmap, result);

	  lr_ignore_rest (ldfile, 1);
	  break;

	case tok_coll_weight_max:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  if (state != 0)
	    goto err_label;

	  arg = lr_token (ldfile, charmap, result, NULL, verbose);
	  if (arg->tok != tok_number)
	    goto err_label;
	  if (collate->col_weight_max != -1)
	    lr_error (ldfile, _("%s: duplicate definition of `%s'"),
		      "LC_COLLATE", "col_weight_max");
	  else
	    collate->col_weight_max = arg->val.num;
	  lr_ignore_rest (ldfile, 1);
	  break;

	case tok_section_symbol:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  if (state != 0)
	    goto err_label;

	  arg = lr_token (ldfile, charmap, result, repertoire, verbose);
	  if (arg->tok != tok_bsymbol)
	    goto err_label;
	  else if (!ignore_content)
	    {
	      /* Check whether this section is already known.  */
	      struct section_list *known = collate->sections;
	      while (known != NULL)
		{
		  if (strcmp (known->name, arg->val.str.startmb) == 0)
		    break;
		  known = known->next;
		}

	      if (known != NULL)
		{
		  lr_error (ldfile,
			    _("%s: duplicate declaration of section `%s'"),
			    "LC_COLLATE", arg->val.str.startmb);
		  free (arg->val.str.startmb);
		}
	      else
		collate->sections = make_seclist_elem (collate,
						       arg->val.str.startmb,
						       collate->sections);

	      lr_ignore_rest (ldfile, known == NULL);
	    }
	  else
	    {
	      free (arg->val.str.startmb);
	      lr_ignore_rest (ldfile, 0);
	    }
	  break;

	case tok_collating_element:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  if (state != 0 && state != 2)
	    goto err_label;

	  arg = lr_token (ldfile, charmap, result, repertoire, verbose);
	  if (arg->tok != tok_bsymbol)
	    goto err_label;
	  else
	    {
	      const char *symbol = arg->val.str.startmb;
	      size_t symbol_len = arg->val.str.lenmb;

	      /* Next the `from' keyword.  */
	      arg = lr_token (ldfile, charmap, result, repertoire, verbose);
	      if (arg->tok != tok_from)
		{
		  free ((char *) symbol);
		  goto err_label;
		}

	      ldfile->return_widestr = 1;
	      ldfile->translate_strings = 1;

	      /* Finally the string with the replacement.  */
	      arg = lr_token (ldfile, charmap, result, repertoire, verbose);

	      ldfile->return_widestr = 0;
	      ldfile->translate_strings = 0;

	      if (arg->tok != tok_string)
		goto err_label;

	      if (!ignore_content && symbol != NULL)
		{
		  /* The name is already defined.  */
		  if (check_duplicate (ldfile, collate, charmap,
				       repertoire, symbol, symbol_len))
		    goto col_elem_free;

		  if (arg->val.str.startmb != NULL)
		    insert_entry (&collate->elem_table, symbol, symbol_len,
				  new_element (collate,
					       arg->val.str.startmb,
					       arg->val.str.lenmb - 1,
					       arg->val.str.startwc,
					       symbol, symbol_len, 0));
		}
	      else
		{
		col_elem_free:
		  free ((char *) symbol);
		  free (arg->val.str.startmb);
		  free (arg->val.str.startwc);
		}
	      lr_ignore_rest (ldfile, 1);
	    }
	  break;

	case tok_collating_symbol:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  if (state != 0 && state != 2)
	    goto err_label;

	  arg = lr_token (ldfile, charmap, result, repertoire, verbose);
	  if (arg->tok != tok_bsymbol)
	    goto err_label;
	  else
	    {
	      char *symbol = arg->val.str.startmb;
	      size_t symbol_len = arg->val.str.lenmb;
	      char *endsymbol = NULL;
	      size_t endsymbol_len = 0;
	      enum token_t ellipsis = tok_none;

	      arg = lr_token (ldfile, charmap, result, repertoire, verbose);
	      if (arg->tok == tok_ellipsis2 || arg->tok == tok_ellipsis4)
		{
		  ellipsis = arg->tok;

		  arg = lr_token (ldfile, charmap, result, repertoire,
				  verbose);
		  if (arg->tok != tok_bsymbol)
		    {
		      free (symbol);
		      goto err_label;
		    }

		  endsymbol = arg->val.str.startmb;
		  endsymbol_len = arg->val.str.lenmb;

		  lr_ignore_rest (ldfile, 1);
		}
	      else if (arg->tok != tok_eol)
		{
		  free (symbol);
		  goto err_label;
		}

	      if (!ignore_content)
		{
		  if (symbol == NULL
		      || (ellipsis != tok_none && endsymbol == NULL))
		    {
		      lr_error (ldfile, _("\
%s: unknown character in collating symbol name"),
				"LC_COLLATE");
		      goto col_sym_free;
		    }
		  else if (ellipsis == tok_none)
		    {
		      /* A single symbol, no ellipsis.  */
		      if (check_duplicate (ldfile, collate, charmap,
					   repertoire, symbol, symbol_len))
			/* The name is already defined.  */
			goto col_sym_free;

		      insert_entry (&collate->sym_table, symbol, symbol_len,
				    new_symbol (collate, symbol, symbol_len));
		    }
		  else if (symbol_len != endsymbol_len)
		    {
		    col_sym_inv_range:
		      lr_error (ldfile,
				_("invalid names for character range"));
		      goto col_sym_free;
		    }
		  else
		    {
		      /* Oh my, we have to handle an ellipsis.  First, as
			 usual, determine the common prefix and then
			 convert the rest into a range.  */
		      size_t prefixlen;
		      unsigned long int from;
		      unsigned long int to;
		      char *endp;

		      for (prefixlen = 0; prefixlen < symbol_len; ++prefixlen)
			if (symbol[prefixlen] != endsymbol[prefixlen])
			  break;

		      /* Convert the rest into numbers.  */
		      symbol[symbol_len] = '\0';
		      from = strtoul (&symbol[prefixlen], &endp,
				      ellipsis == tok_ellipsis2 ? 16 : 10);
		      if (*endp != '\0')
			goto col_sym_inv_range;

		      endsymbol[symbol_len] = '\0';
		      to = strtoul (&endsymbol[prefixlen], &endp,
				    ellipsis == tok_ellipsis2 ? 16 : 10);
		      if (*endp != '\0')
			goto col_sym_inv_range;

		      if (from > to)
			goto col_sym_inv_range;

		      /* Now loop over all entries.  */
		      while (from <= to)
			{
			  char *symbuf;

			  symbuf = (char *) obstack_alloc (&collate->mempool,
							   symbol_len + 1);

			  /* Create the name.  */
			  sprintf (symbuf,
				   ellipsis == tok_ellipsis2
				   ? "%.*s%.*lX" : "%.*s%.*lu",
				   (int) prefixlen, symbol,
				   (int) (symbol_len - prefixlen), from);

			  if (check_duplicate (ldfile, collate, charmap,
					       repertoire, symbuf, symbol_len))
			    /* The name is already defined.  */
			    goto col_sym_free;

			  insert_entry (&collate->sym_table, symbuf,
					symbol_len,
					new_symbol (collate, symbuf,
						    symbol_len));

			  /* Increment the counter.  */
			  ++from;
			}

		      goto col_sym_free;
		    }
		}
	      else
		{
		col_sym_free:
		  free (symbol);
		  free (endsymbol);
		}
	    }
	  break;

	case tok_symbol_equivalence:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  if (state != 0)
	    goto err_label;

	  arg = lr_token (ldfile, charmap, result, repertoire, verbose);
	  if (arg->tok != tok_bsymbol)
	    goto err_label;
	  else
	    {
	      const char *newname = arg->val.str.startmb;
	      size_t newname_len = arg->val.str.lenmb;
	      const char *symname;
	      size_t symname_len;
	      void *symval;	/* Actually struct symbol_t*  */

	      arg = lr_token (ldfile, charmap, result, repertoire, verbose);
	      if (arg->tok != tok_bsymbol)
		{
		  free ((char *) newname);
		  goto err_label;
		}

	      symname = arg->val.str.startmb;
	      symname_len = arg->val.str.lenmb;

	      if (newname == NULL)
		{
		  lr_error (ldfile, _("\
%s: unknown character in equivalent definition name"),
			    "LC_COLLATE");

		sym_equiv_free:
		  free ((char *) newname);
		  free ((char *) symname);
		  break;
		}
	      if (symname == NULL)
		{
		  lr_error (ldfile, _("\
%s: unknown character in equivalent definition value"),
			    "LC_COLLATE");
		  goto sym_equiv_free;
		}

	      /* See whether the symbol name is already defined.  */
	      if (find_entry (&collate->sym_table, symname, symname_len,
			      &symval) != 0)
		{
		  lr_error (ldfile, _("\
%s: unknown symbol `%s' in equivalent definition"),
			    "LC_COLLATE", symname);
		  goto sym_equiv_free;
		}

	      if (insert_entry (&collate->sym_table,
				newname, newname_len, symval) < 0)
		{
		  lr_error (ldfile, _("\
error while adding equivalent collating symbol"));
		  goto sym_equiv_free;
		}

	      free ((char *) symname);
	    }
	  lr_ignore_rest (ldfile, 1);
	  break;

	case tok_script:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  /* We get told about the scripts we know.  */
	  arg = lr_token (ldfile, charmap, result, repertoire, verbose);
	  if (arg->tok != tok_bsymbol)
	    goto err_label;
	  else
	    {
	      struct section_list *runp = collate->known_sections;
	      char *name;

	      while (runp != NULL)
		if (strncmp (runp->name, arg->val.str.startmb,
			     arg->val.str.lenmb) == 0
		    && runp->name[arg->val.str.lenmb] == '\0')
		  break;
		else
		  runp = runp->def_next;

	      if (runp != NULL)
		{
		  lr_error (ldfile, _("duplicate definition of script `%s'"),
			    runp->name);
		  lr_ignore_rest (ldfile, 0);
		  break;
		}

	      runp = (struct section_list *) xcalloc (1, sizeof (*runp));
	      name = (char *) xmalloc (arg->val.str.lenmb + 1);
	      memcpy (name, arg->val.str.startmb, arg->val.str.lenmb);
	      name[arg->val.str.lenmb] = '\0';
	      runp->name = name;

	      runp->def_next = collate->known_sections;
	      collate->known_sections = runp;
	    }
	  lr_ignore_rest (ldfile, 1);
	  break;

	case tok_order_start:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  if (state != 0 && state != 1 && state != 2)
	    goto err_label;
	  state = 1;

	  /* The 14652 draft does not specify whether all `order_start' lines
	     must contain the same number of sort-rules, but 14651 does.  So
	     we require this here as well.  */
	  arg = lr_token (ldfile, charmap, result, repertoire, verbose);
	  if (arg->tok == tok_bsymbol)
	    {
	      /* This better should be a section name.  */
	      struct section_list *sp = collate->known_sections;
	      while (sp != NULL
		     && (sp->name == NULL
			 || strncmp (sp->name, arg->val.str.startmb,
				     arg->val.str.lenmb) != 0
			 || sp->name[arg->val.str.lenmb] != '\0'))
		sp = sp->def_next;

	      if (sp == NULL)
		{
		  lr_error (ldfile, _("\
%s: unknown section name `%.*s'"),
			    "LC_COLLATE", (int) arg->val.str.lenmb,
			    arg->val.str.startmb);
		  /* We use the error section.  */
		  collate->current_section = &collate->error_section;

		  if (collate->error_section.first == NULL)
		    {
		      /* Insert &collate->error_section at the end of
			 the collate->sections list.  */
		      if (collate->sections == NULL)
			collate->sections = &collate->error_section;
		      else
			{
			  sp = collate->sections;
			  while (sp->next != NULL)
			    sp = sp->next;

			  sp->next = &collate->error_section;
			}
		      collate->error_section.next = NULL;
		    }
		}
	      else
		{
		  /* One should not be allowed to open the same
		     section twice.  */
		  if (sp->first != NULL)
		    lr_error (ldfile, _("\
%s: multiple order definitions for section `%s'"),
			      "LC_COLLATE", sp->name);
		  else
		    {
		      /* Insert sp in the collate->sections list,
			 right after collate->current_section.  */
		      if (collate->current_section != NULL)
			{
			  sp->next = collate->current_section->next;
			  collate->current_section->next = sp;
			}
		      else if (collate->sections == NULL)
			/* This is the first section to be defined.  */
			collate->sections = sp;

		      collate->current_section = sp;
		    }

		  /* Next should come the end of the line or a semicolon.  */
		  arg = lr_token (ldfile, charmap, result, repertoire,
				  verbose);
		  if (arg->tok == tok_eol)
		    {
		      uint32_t cnt;

		      /* This means we have exactly one rule: `forward'.  */
		      if (nrules > 1)
			lr_error (ldfile, _("\
%s: invalid number of sorting rules"),
				  "LC_COLLATE");
		      else
			nrules = 1;
		      sp->rules = obstack_alloc (&collate->mempool,
						 (sizeof (enum coll_sort_rule)
						  * nrules));
		      for (cnt = 0; cnt < nrules; ++cnt)
			sp->rules[cnt] = sort_forward;

		      /* Next line.  */
		      break;
		    }

		  /* Get the next token.  */
		  arg = lr_token (ldfile, charmap, result, repertoire,
				  verbose);
		}
	    }
	  else
	    {
	      /* There is no section symbol.  Therefore we use the unnamed
		 section.  */
	      collate->current_section = &collate->unnamed_section;

	      if (collate->unnamed_section_defined)
		lr_error (ldfile, _("\
%s: multiple order definitions for unnamed section"),
			  "LC_COLLATE");
	      else
		{
		  /* Insert &collate->unnamed_section at the beginning of
		     the collate->sections list.  */
		  collate->unnamed_section.next = collate->sections;
		  collate->sections = &collate->unnamed_section;
		  collate->unnamed_section_defined = true;
		}
	    }

	  /* Now read the direction names.  */
	  read_directions (ldfile, arg, charmap, repertoire, result);

	  /* From now we need the strings untranslated.  */
	  ldfile->translate_strings = 0;
	  break;

	case tok_order_end:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  if (state != 1)
	    goto err_label;

	  /* Handle ellipsis at end of list.  */
	  if (was_ellipsis != tok_none)
	    {
	      handle_ellipsis (ldfile, NULL, 0, was_ellipsis, charmap,
			       repertoire, result);
	      was_ellipsis = tok_none;
	    }

	  state = 2;
	  lr_ignore_rest (ldfile, 1);
	  break;

	case tok_reorder_after:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  if (state == 1)
	    {
	      lr_error (ldfile, _("%s: missing `order_end' keyword"),
			"LC_COLLATE");
	      state = 2;

	      /* Handle ellipsis at end of list.  */
	      if (was_ellipsis != tok_none)
		{
		  handle_ellipsis (ldfile, arg->val.str.startmb,
				   arg->val.str.lenmb, was_ellipsis, charmap,
				   repertoire, result);
		  was_ellipsis = tok_none;
		}
	    }
	  else if (state == 0 && copy_locale == NULL)
	    goto err_label;
	  else if (state != 0 && state != 2 && state != 3)
	    goto err_label;
	  state = 3;

	  arg = lr_token (ldfile, charmap, result, repertoire, verbose);
	  if (arg->tok == tok_bsymbol || arg->tok == tok_ucs4)
	    {
	      /* Find this symbol in the sequence table.  */
	      char ucsbuf[10];
	      char *startmb;
	      size_t lenmb;
	      struct element_t *insp;
	      int no_error = 1;
	      void *ptr;

	      if (arg->tok == tok_bsymbol)
		{
		  startmb = arg->val.str.startmb;
		  lenmb = arg->val.str.lenmb;
		}
	      else
		{
		  sprintf (ucsbuf, "U%08X", arg->val.ucs4);
		  startmb = ucsbuf;
		  lenmb = 9;
		}

	      if (find_entry (&collate->seq_table, startmb, lenmb, &ptr) == 0)
		/* Yes, the symbol exists.  Simply point the cursor
		   to it.  */
		collate->cursor = (struct element_t *) ptr;
	      else
		{
		  struct symbol_t *symbp;
		  void *ptr;

		  if (find_entry (&collate->sym_table, startmb, lenmb,
				  &ptr) == 0)
		    {
		      symbp = ptr;

		      if (symbp->order->last != NULL
			  || symbp->order->next != NULL)
			collate->cursor = symbp->order;
		      else
			{
			  /* This is a collating symbol but its position
			     is not yet defined.  */
			  lr_error (ldfile, _("\
%s: order for collating symbol %.*s not yet defined"),
				    "LC_COLLATE", (int) lenmb, startmb);
			  collate->cursor = NULL;
			  no_error = 0;
			}
		    }
		  else if (find_entry (&collate->elem_table, startmb, lenmb,
				       &ptr) == 0)
		    {
		      insp = (struct element_t *) ptr;

		      if (insp->last != NULL || insp->next != NULL)
			collate->cursor = insp;
		      else
			{
			  /* This is a collating element but its position
			     is not yet defined.  */
			  lr_error (ldfile, _("\
%s: order for collating element %.*s not yet defined"),
				    "LC_COLLATE", (int) lenmb, startmb);
			  collate->cursor = NULL;
			  no_error = 0;
			}
		    }
		  else
		    {
		      /* This is bad.  The symbol after which we have to
			 insert does not exist.  */
		      lr_error (ldfile, _("\
%s: cannot reorder after %.*s: symbol not known"),
				"LC_COLLATE", (int) lenmb, startmb);
		      collate->cursor = NULL;
		      no_error = 0;
		    }
		}

	      lr_ignore_rest (ldfile, no_error);
	    }
	  else
	    /* This must not happen.  */
	    goto err_label;
	  break;

	case tok_reorder_end:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    break;

	  if (state != 3)
	    goto err_label;
	  state = 4;
	  lr_ignore_rest (ldfile, 1);
	  break;

	case tok_reorder_sections_after:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  if (state == 1)
	    {
	      lr_error (ldfile, _("%s: missing `order_end' keyword"),
			"LC_COLLATE");
	      state = 2;

	      /* Handle ellipsis at end of list.  */
	      if (was_ellipsis != tok_none)
		{
		  handle_ellipsis (ldfile, NULL, 0, was_ellipsis, charmap,
				   repertoire, result);
		  was_ellipsis = tok_none;
		}
	    }
	  else if (state == 3)
	    {
	      record_error (0, 0, _("\
%s: missing `reorder-end' keyword"), "LC_COLLATE");
	      state = 4;
	    }
	  else if (state != 2 && state != 4)
	    goto err_label;
	  state = 5;

	  /* Get the name of the sections we are adding after.  */
	  arg = lr_token (ldfile, charmap, result, repertoire, verbose);
	  if (arg->tok == tok_bsymbol)
	    {
	      /* Now find a section with this name.  */
	      struct section_list *runp = collate->sections;

	      while (runp != NULL)
		{
		  if (runp->name != NULL
		      && strlen (runp->name) == arg->val.str.lenmb
		      && memcmp (runp->name, arg->val.str.startmb,
				 arg->val.str.lenmb) == 0)
		    break;

		  runp = runp->next;
		}

	      if (runp != NULL)
		collate->current_section = runp;
	      else
		{
		  /* This is bad.  The section after which we have to
		     reorder does not exist.  Therefore we cannot
		     process the whole rest of this reorder
		     specification.  */
		  lr_error (ldfile, _("%s: section `%.*s' not known"),
			    "LC_COLLATE", (int) arg->val.str.lenmb,
			    arg->val.str.startmb);

		  do
		    {
		      lr_ignore_rest (ldfile, 0);

		      now = lr_token (ldfile, charmap, result, NULL, verbose);
		    }
		  while (now->tok == tok_reorder_sections_after
			 || now->tok == tok_reorder_sections_end
			 || now->tok == tok_end);

		  /* Process the token we just saw.  */
		  nowtok = now->tok;
		  continue;
		}
	    }
	  else
	    /* This must not happen.  */
	    goto err_label;
	  break;

	case tok_reorder_sections_end:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    break;

	  if (state != 5)
	    goto err_label;
	  state = 6;
	  lr_ignore_rest (ldfile, 1);
	  break;

	case tok_bsymbol:
	case tok_ucs4:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  if (state != 0 && state != 1 && state != 3 && state != 5)
	    goto err_label;

	  if ((state == 0 || state == 5) && nowtok == tok_ucs4)
	    goto err_label;

	  if (nowtok == tok_ucs4)
	    {
	      snprintf (ucs4buf, sizeof (ucs4buf), "U%08X", now->val.ucs4);
	      symstr = ucs4buf;
	      symlen = 9;
	    }
	  else if (arg != NULL)
	    {
	      symstr = arg->val.str.startmb;
	      symlen = arg->val.str.lenmb;
	    }
	  else
	    {
	      lr_error (ldfile, _("%s: bad symbol <%.*s>"), "LC_COLLATE",
			(int) ldfile->token.val.str.lenmb,
			ldfile->token.val.str.startmb);
	      break;
	    }

	  struct element_t *seqp;
	  if (state == 0)
	    {
	      /* We are outside an `order_start' region.  This means
		 we must only accept definitions of values for
		 collation symbols since these are purely abstract
		 values and don't need directions associated.  */
	      void *ptr;

	      if (find_entry (&collate->seq_table, symstr, symlen, &ptr) == 0)
		{
		  seqp = ptr;

		  /* It's already defined.  First check whether this
		     is really a collating symbol.  */
		  if (seqp->is_character)
		    goto err_label;

		  goto move_entry;
		}
	      else
		{
		  void *result;

		  if (find_entry (&collate->sym_table, symstr, symlen,
				  &result) != 0)
		    /* No collating symbol, it's an error.  */
		    goto err_label;

		  /* Maybe this is the first time we define a symbol
		     value and it is before the first actual section.  */
		  if (collate->sections == NULL)
		    collate->sections = collate->current_section =
		      &collate->symbol_section;
		}

	      if (was_ellipsis != tok_none)
		{
		  handle_ellipsis (ldfile, symstr, symlen, was_ellipsis,
				   charmap, repertoire, result);

		  /* Remember that we processed the ellipsis.  */
		  was_ellipsis = tok_none;

		  /* And don't add the value a second time.  */
		  break;
		}
	    }
	  else if (state == 3)
	    {
	      /* It is possible that we already have this collation sequence.
		 In this case we move the entry.  */
	      void *sym;
	      void *ptr;

	      /* If the symbol after which we have to insert was not found
		 ignore all entries.  */
	      if (collate->cursor == NULL)
		{
		  lr_ignore_rest (ldfile, 0);
		  break;
		}

	      if (find_entry (&collate->seq_table, symstr, symlen, &ptr) == 0)
		{
		  seqp = (struct element_t *) ptr;
		  goto move_entry;
		}

	      if (find_entry (&collate->sym_table, symstr, symlen, &sym) == 0
		  && (seqp = ((struct symbol_t *) sym)->order) != NULL)
		goto move_entry;

	      if (find_entry (&collate->elem_table, symstr, symlen, &ptr) == 0
		  && (seqp = (struct element_t *) ptr,
		      seqp->last != NULL || seqp->next != NULL
		      || (collate->start != NULL && seqp == collate->start)))
		{
		move_entry:
		  /* Remove the entry from the old position.  */
		  if (seqp->last == NULL)
		    collate->start = seqp->next;
		  else
		    seqp->last->next = seqp->next;
		  if (seqp->next != NULL)
		    seqp->next->last = seqp->last;

		  /* We also have to check whether this entry is the
		     first or last of a section.  */
		  if (seqp->section->first == seqp)
		    {
		      if (seqp->section->first == seqp->section->last)
			/* This section has no content anymore.  */
			seqp->section->first = seqp->section->last = NULL;
		      else
			seqp->section->first = seqp->next;
		    }
		  else if (seqp->section->last == seqp)
		    seqp->section->last = seqp->last;

		  /* Now insert it in the new place.  */
		  insert_weights (ldfile, seqp, charmap, repertoire, result,
				  tok_none);
		  break;
		}

	      /* Otherwise we just add a new entry.  */
	    }
	  else if (state == 5)
	    {
	      /* We are reordering sections.  Find the named section.  */
	      struct section_list *runp = collate->sections;
	      struct section_list *prevp = NULL;

	      while (runp != NULL)
		{
		  if (runp->name != NULL
		      && strlen (runp->name) == symlen
		      && memcmp (runp->name, symstr, symlen) == 0)
		    break;

		  prevp = runp;
		  runp = runp->next;
		}

	      if (runp == NULL)
		{
		  lr_error (ldfile, _("%s: section `%.*s' not known"),
			    "LC_COLLATE", (int) symlen, symstr);
		  lr_ignore_rest (ldfile, 0);
		}
	      else
		{
		  if (runp != collate->current_section)
		    {
		      /* Remove the named section from the old place and
			 insert it in the new one.  */
		      prevp->next = runp->next;

		      runp->next = collate->current_section->next;
		      collate->current_section->next = runp;
		      collate->current_section = runp;
		    }

		  /* Process the rest of the line which might change
		     the collation rules.  */
		  arg = lr_token (ldfile, charmap, result, repertoire,
				  verbose);
		  if (arg->tok != tok_eof && arg->tok != tok_eol)
		    read_directions (ldfile, arg, charmap, repertoire,
				     result);
		}
	      break;
	    }
	  else if (was_ellipsis != tok_none)
	    {
	      /* Using the information in the `ellipsis_weight'
		 element and this and the last value we have to handle
		 the ellipsis now.  */
	      assert (state == 1);

	      handle_ellipsis (ldfile, symstr, symlen, was_ellipsis, charmap,
			       repertoire, result);

	      /* Remember that we processed the ellipsis.  */
	      was_ellipsis = tok_none;

	      /* And don't add the value a second time.  */
	      break;
	    }

	  /* Now insert in the new place.  */
	  insert_value (ldfile, symstr, symlen, charmap, repertoire, result);
	  break;

	case tok_undefined:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  if (state != 1)
	    goto err_label;

	  if (was_ellipsis != tok_none)
	    {
	      lr_error (ldfile,
			_("%s: cannot have `%s' as end of ellipsis range"),
			"LC_COLLATE", "UNDEFINED");

	      unlink_element (collate);
	      was_ellipsis = tok_none;
	    }

	  /* See whether UNDEFINED already appeared somewhere.  */
	  if (collate->undefined.next != NULL
	      || &collate->undefined == collate->cursor)
	    {
	      lr_error (ldfile,
			_("%s: order for `%.*s' already defined at %s:%zu"),
			"LC_COLLATE", 9, "UNDEFINED",
			collate->undefined.file,
			collate->undefined.line);
	      lr_ignore_rest (ldfile, 0);
	    }
	  else
	    /* Parse the weights.  */
	     insert_weights (ldfile, &collate->undefined, charmap,
			     repertoire, result, tok_none);
	  break;

	case tok_ellipsis2: /* symbolic hexadecimal ellipsis */
	case tok_ellipsis3: /* absolute ellipsis */
	case tok_ellipsis4: /* symbolic decimal ellipsis */
	  /* This is the symbolic (decimal or hexadecimal) or absolute
	     ellipsis.  */
	  if (was_ellipsis != tok_none)
	    goto err_label;

	  if (state != 0 && state != 1 && state != 3)
	    goto err_label;

	  was_ellipsis = nowtok;

	  insert_weights (ldfile, &collate->ellipsis_weight, charmap,
			  repertoire, result, nowtok);
	  break;

	case tok_end:
	seen_end:
	  /* Next we assume `LC_COLLATE'.  */
	  if (!ignore_content)
	    {
	      if (state == 0 && copy_locale == NULL)
		/* We must either see a copy statement or have
		   ordering values.  */
		lr_error (ldfile,
			  _("%s: empty category description not allowed"),
			  "LC_COLLATE");
	      else if (state == 1)
		{
		  lr_error (ldfile, _("%s: missing `order_end' keyword"),
			    "LC_COLLATE");

		  /* Handle ellipsis at end of list.  */
		  if (was_ellipsis != tok_none)
		    {
		      handle_ellipsis (ldfile, NULL, 0, was_ellipsis, charmap,
				       repertoire, result);
		      was_ellipsis = tok_none;
		    }
		}
	      else if (state == 3)
		record_error (0, 0, _("\
%s: missing `reorder-end' keyword"), "LC_COLLATE");
	      else if (state == 5)
		record_error (0, 0, _("\
%s: missing `reorder-sections-end' keyword"), "LC_COLLATE");
	    }
	  arg = lr_token (ldfile, charmap, result, NULL, verbose);
	  if (arg->tok == tok_eof)
	    break;
	  if (arg->tok == tok_eol)
	    lr_error (ldfile, _("%s: incomplete `END' line"), "LC_COLLATE");
	  else if (arg->tok != tok_lc_collate)
	    lr_error (ldfile, _("\
%1$s: definition does not end with `END %1$s'"), "LC_COLLATE");
	  lr_ignore_rest (ldfile, arg->tok == tok_lc_collate);
	  return;

	case tok_define:
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  arg = lr_token (ldfile, charmap, result, NULL, verbose);
	  if (arg->tok != tok_ident)
	    goto err_label;

	  /* Simply add the new symbol.  */
	  struct name_list *newsym = xmalloc (sizeof (*newsym)
					      + arg->val.str.lenmb + 1);
	  memcpy (newsym->str, arg->val.str.startmb, arg->val.str.lenmb);
	  newsym->str[arg->val.str.lenmb] = '\0';
	  newsym->next = defined;
	  defined = newsym;

	  lr_ignore_rest (ldfile, 1);
	  break;

	case tok_undef:
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  arg = lr_token (ldfile, charmap, result, NULL, verbose);
	  if (arg->tok != tok_ident)
	    goto err_label;

	  /* Remove _all_ occurrences of the symbol from the list.  */
	  struct name_list *prevdef = NULL;
	  struct name_list *curdef = defined;
	  while (curdef != NULL)
	    if (strncmp (arg->val.str.startmb, curdef->str,
			 arg->val.str.lenmb) == 0
		&& curdef->str[arg->val.str.lenmb] == '\0')
	      {
		if (prevdef == NULL)
		  defined = curdef->next;
		else
		  prevdef->next = curdef->next;

		struct name_list *olddef = curdef;
		curdef = curdef->next;

		free (olddef);
	      }
	    else
	      {
		prevdef = curdef;
		curdef = curdef->next;
	      }

	  lr_ignore_rest (ldfile, 1);
	  break;

	case tok_ifdef:
	case tok_ifndef:
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	found_ifdef:
	  arg = lr_token (ldfile, charmap, result, NULL, verbose);
	  if (arg->tok != tok_ident)
	    goto err_label;
	  lr_ignore_rest (ldfile, 1);

	  if (collate->else_action == else_none)
	    {
	      curdef = defined;
	      while (curdef != NULL)
		if (strncmp (arg->val.str.startmb, curdef->str,
			     arg->val.str.lenmb) == 0
		    && curdef->str[arg->val.str.lenmb] == '\0')
		  break;
		else
		  curdef = curdef->next;

	      if ((nowtok == tok_ifdef && curdef != NULL)
		  || (nowtok == tok_ifndef && curdef == NULL))
		{
		  /* We have to use the if-branch.  */
		  collate->else_action = else_ignore;
		}
	      else
		{
		  /* We have to use the else-branch, if there is one.  */
		  nowtok = skip_to (ldfile, collate, charmap, 0);
		  if (nowtok == tok_else)
		    collate->else_action = else_seen;
		  else if (nowtok == tok_elifdef)
		    {
		      nowtok = tok_ifdef;
		      goto found_ifdef;
		    }
		  else if (nowtok == tok_elifndef)
		    {
		      nowtok = tok_ifndef;
		      goto found_ifdef;
		    }
		  else if (nowtok == tok_eof)
		    goto seen_eof;
		  else if (nowtok == tok_end)
		    goto seen_end;
		}
	    }
	  else
	    {
	      /* XXX Should it really become necessary to support nested
		 preprocessor handling we will push the state here.  */
	      lr_error (ldfile, _("%s: nested conditionals not supported"),
			"LC_COLLATE");
	      nowtok = skip_to (ldfile, collate, charmap, 1);
	      if (nowtok == tok_eof)
		goto seen_eof;
	      else if (nowtok == tok_end)
		goto seen_end;
	    }
	  break;

	case tok_elifdef:
	case tok_elifndef:
	case tok_else:
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  lr_ignore_rest (ldfile, 1);

	  if (collate->else_action == else_ignore)
	    {
	      /* Ignore everything until the endif.  */
	      nowtok = skip_to (ldfile, collate, charmap, 1);
	      if (nowtok == tok_eof)
		goto seen_eof;
	      else if (nowtok == tok_end)
		goto seen_end;
	    }
	  else
	    {
	      assert (collate->else_action == else_none);
	      lr_error (ldfile, _("\
%s: '%s' without matching 'ifdef' or 'ifndef'"), "LC_COLLATE",
			nowtok == tok_else ? "else"
			: nowtok == tok_elifdef ? "elifdef" : "elifndef");
	    }
	  break;

	case tok_endif:
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  lr_ignore_rest (ldfile, 1);

	  if (collate->else_action != else_ignore
	      && collate->else_action != else_seen)
	    lr_error (ldfile, _("\
%s: 'endif' without matching 'ifdef' or 'ifndef'"), "LC_COLLATE");

	  /* XXX If we support nested preprocessor directives we pop
	     the state here.  */
	  collate->else_action = else_none;
	  break;

	default:
	err_label:
	  SYNTAX_ERROR (_("%s: syntax error"), "LC_COLLATE");
	}

      /* Prepare for the next round.  */
      now = lr_token (ldfile, charmap, result, NULL, verbose);
      nowtok = now->tok;
    }

 seen_eof:
  /* When we come here we reached the end of the file.  */
  lr_error (ldfile, _("%s: premature end of file"), "LC_COLLATE");
}
