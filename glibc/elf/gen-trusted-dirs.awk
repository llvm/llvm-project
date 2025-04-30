BEGIN {
  FS = " ";
}

{
  for (i = 1; i <= NF; ++i) {
    s[cnt++] = $i"/";
  }
}

END {
  printf ("#define SYSTEM_DIRS \\\n");

  printf ("  \"%s\"", s[0]);

  for (i = 1; i < cnt; ++i) {
    printf (" \"\\0\" \"%s\"", s[i]);
  }

  printf ("\n\n");

  printf ("#define SYSTEM_DIRS_LEN \\\n");

  printf ("  %d", length (s[0]));
  m = length (s[0]);

  for (i = 1; i < cnt; ++i) {
    printf (", %d", length(s[i]));
    if (length(s[i]) > m) {
      m = length(s[i]);
    }
  }

  printf ("\n\n");

  printf ("#define SYSTEM_DIRS_MAX_LEN\t%d\n", m);
}
