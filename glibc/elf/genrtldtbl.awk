#!/usr/bin/awk
BEGIN {
  FS=":";
  count=0;
}
{
  for (i = 1; i <= NF; ++i) {
    gsub (/\/*$/, "", $i);
    dir[count++] = $i;
  }
}
END {
  for (i = 0; i < count; ++i) {
    printf ("static struct r_search_path_elem rtld_search_dir%d =\n", i+1);
    printf ("  { \"%s/\", %d, unknown, 0, nonexisting, NULL, NULL, ",
	    dir[i], length (dir[i]) + 1);
    if (i== 0)
      printf ("NULL };\n");
    else
      printf ("&rtld_search_dir%d };\n", i);
  }
  printf ("\nstatic struct r_search_path_elem *rtld_search_dirs[] =\n{\n");
  for (i = 0; i < count; ++i) {
    printf ("  &rtld_search_dir%d,\n", i + 1);
  }
  printf ("  NULL\n};\n\n");
  printf ("static struct r_search_path_elem *all_dirs = &rtld_search_dir%d;\n",
	  count);
}
