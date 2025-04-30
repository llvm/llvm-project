#!/usr/bin/awk -f
# Generate sorted list of directories.  The sorting is stable but with
# dependencies between directories resolved by moving dependees in front.
# Copyright (C) 1998-2021 Free Software Foundation, Inc.
# Written by Ulrich Drepper <drepper@cygnus.com>, 1998.

BEGIN {
  cnt = split(subdirs, all) + 1
  dnt = 0
}

# Let input files have comments.
{ sub(/[ 	]*#.*$/, "") }
NF == 0 { next }

{
  subdir = type = FILENAME;
  sub(/^.*\//, "", type);
  sub(/\/[^/]+$/, "", subdir);
  sub(/^.*\//, "", subdir);
  thisdir = "";
}

type == "Depend" && NF == 1 {
  from[dnt] = subdir;
  to[dnt] = $1;
  ++dnt;
  next
}

type == "Subdirs" && NF == 1 { thisdir = $1 }

type == "Subdirs" && NF == 2 && $1 == "first" {
  thisdir = $2;
  # Make the first dir in the list depend on this one.
  from[dnt] = all[1];
  to[dnt] = thisdir;
  ++dnt;
}

type == "Subdirs" && NF == 2 && $1 == "inhibit" {
  inhibit[$2] = subdir;
  next
}

type == "Subdirs" && thisdir {
  all[cnt++] = thisdir;

  this_srcdir = srcpfx thisdir
  if (system("test -d " this_srcdir) != 0) {
    print FILENAME ":" FNR ":", "cannot find", this_srcdir > "/dev/stderr";
    exit 2
  }
  file = this_srcdir "/Depend";
  if (system("test -f " file) == 0) {
    ARGV[ARGC++] = file;
    # Emit a dependency on the implicitly-read file.
    if (srcpfx)
      sub(/^\.\.\//, "", file);
    if (file !~ /^\/.*$/)
      file = "$(..)" file;
    print "$(common-objpfx)sysd-sorted:", "$(wildcard", file ")";
  }
  next
}

{
  print FILENAME ":" FNR ":", "what type of file is this?" > "/dev/stderr";
  exit 2
}

END {
  do {
    moved = 0
    for (i = 0; i < dnt; ++i) {
      for (j = 1; j < cnt; ++j) {
	if (all[j] == from[i]) {
	  for (k = j + 1; k < cnt; ++k) {
	    if (all[k] == to[i]) {
	      break;
	    }
	  }
	  if (k < cnt) {
	    for (l = k - 1; l >= j; --l) {
	      all[l + 1] = all[l]
	    }
	    all[j] = to[i]
	    break;
	  }
	}
      }
      if (j < cnt) {
	moved = 1
	break
      }
    }
  } while (moved);

  # Make sure we list "elf" last.
  saw_elf = 0;
  printf "sorted-subdirs :=";
  for (i = 1; i < cnt; ++i) {
    if (all[i] in inhibit)
      continue;
    if (all[i] == "elf")
      saw_elf = 1;
    else
      printf " %s", all[i];
  }
  printf "%s\n", saw_elf ? " elf" : "";

  print "sysd-sorted-done := t"
}
