/* Reproduce a GNU malloc bug.  */
#include <malloc.h>
#include <stdio.h>
#include <string.h>

#define size_t unsigned int

/* Defined as global variables to avoid warnings about unused variables.  */
char *dummy0;
char *dummy1;
char *fill_info_table1;


int
main (int argc, char *argv[])
{
  char *over_top;
  size_t over_top_size = 0x3000;
  char *over_top_dup;
  size_t over_top_dup_size = 0x7000;
  char *x;
  size_t i;

  /* Here's what memory is supposed to look like (hex):
        size  contents
        3000  original_info_table, later fill_info_table1
      3fa000  dummy0
      3fa000  dummy1
        6000  info_table_2
        3000  over_top

   */
  /* mem: original_info_table */
  dummy0 = malloc (0x3fa000);
  /* mem: original_info_table, dummy0 */
  dummy1 = malloc (0x3fa000);
  /* mem: free, dummy0, dummy1, info_table_2 */
  fill_info_table1 = malloc (0x3000);
  /* mem: fill_info_table1, dummy0, dummy1, info_table_2 */

  x = malloc (0x1000);
  free (x);
  /* mem: fill_info_table1, dummy0, dummy1, info_table_2, freexx */

  /* This is what loses; info_table_2 and freexx get combined unbeknownst
     to mmalloc, and mmalloc puts over_top in a section of memory which
     is on the free list as part of another block (where info_table_2 had
     been).  */
  over_top = malloc (over_top_size);
  over_top_dup = malloc (over_top_dup_size);
  memset (over_top, 0, over_top_size);
  memset (over_top_dup, 1, over_top_dup_size);

  for (i = 0; i < over_top_size; ++i)
    if (over_top[i] != 0)
      {
        printf ("FAIL: malloc expands info table\n");
        return 0;
      }

  for (i = 0; i < over_top_dup_size; ++i)
    if (over_top_dup[i] != 1)
      {
        printf ("FAIL: malloc expands info table\n");
        return 0;
      }

  printf ("PASS: malloc expands info table\n");
  return 0;
}
