# Extract ordered list of version sets from Versions files.
# Copyright (C) 2014-2021 Free Software Foundation, Inc.

BEGIN { in_lib = ""; in_version = 0 }

!in_lib && NF == 2 && $2 == "{" {
  in_lib = $1;
  all_libs[in_lib] = 1;
  next
}
!in_lib { next }

NF == 2 && $2 == "{" {
  in_version = 1;
  lib_versions[in_lib, $1] = 1;
  # Partition the version sets into GLIBC_* and others.
  if ($1 ~ /GLIBC_/) {
    libs[in_lib] = libs[in_lib] "  " $1 "\n";
    all_versions[$1] = 1;
  }
  else {
    others_libs[in_lib] = others_libs[in_lib] "  " $1 "\n";
    others_all_versions[$1] = 1;
  }
  next
}

in_version && $1 == "}" { in_version = 0; next }
in_version { next }

$1 == "}" { in_lib = ""; next }

END {
  nlibs = asorti(all_libs, libs_order);
  for (i = 1; i <= nlibs; ++i) {
    lib = libs_order[i];

    for (v in all_versions) {
      if (!((lib, v) in lib_versions)) {
        libs[lib] = libs[lib] "  " v "\n";
      }
    }

    for (v in others_all_versions) {
      if (!((lib, v) in lib_versions)) {
        others_libs[lib] = others_libs[lib] "  " v "\n";
      }
    }

    print lib, "{";

    # Sort and print all the GLIBC_* sets first, then all the others.
    # This is not really generically right, but it suffices
    # for the cases we have so far.  e.g. GCC_3.0 is "later than"
    # all GLIBC_* sets that matter for purposes of Versions files.

    sort = "sort -u -t. -k 1,1 -k 2n,2n -k 3";
    printf "%s", libs[lib] | sort;
    close(sort);

    sort = "sort -u -t. -k 1,1 -k 2n,2n -k 3";
    printf "%s", others_libs[lib] | sort;
    close(sort);

    print "}";
  }
}
