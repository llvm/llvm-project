#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <locale.h>
#include <locale/localeinfo.h>

int
main (void)
{
  int32_t table_size, idx, i, found;
  const int32_t *symb_table;
  const unsigned char *extra;
  uint32_t nrules;
  char *ca;
  union locale_data_value u;

  ca = setlocale (LC_ALL, "da_DK.ISO-8859-1");
  if (ca == NULL)
    {
      printf ("cannot set locale: %m\n");
      return 1;
    }
  printf ("current locale : %s\n", ca);

  u.string = nl_langinfo (_NL_COLLATE_NRULES);
  nrules = u.word;
  if (nrules == 0)
    {
      printf("No rule\n");
      return 1;
    }

  u.string = nl_langinfo (_NL_COLLATE_SYMB_HASH_SIZEMB);
  table_size = u.word;
  symb_table = (const int32_t *) nl_langinfo (_NL_COLLATE_SYMB_TABLEMB);
  extra = (const unsigned char *) nl_langinfo (_NL_COLLATE_SYMB_EXTRAMB);

  found = 0;
  for (i = 0; i < table_size; ++i)
    {
      if (symb_table[2 * i] != 0)
	{
	  char elem[256];
	  idx = symb_table[2 * i + 1];
	  strncpy (elem, (const char *) (extra + idx + 1), extra[idx]);
	  elem[extra[idx]] = '\0';
	  printf ("Found a collating element: %s\n", elem);
	  ++found;
	}
    }
  if (found == 0)
    {
      printf ("No collating element!\n");
      return 1;
    }
  else if (found != 6)
    {
      printf ("expected 6 collating elements, found %d\n", found);
      return 1;
    }

  return 0;
}
