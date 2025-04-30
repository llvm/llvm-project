# This script reads the contents of Versions.all and outputs a definition
# of variable have-VERSION for each symbol version VERSION which is
# defined.
#
# The have-VERSION variables can be used to check that a port supports a
# particular symbol version in makefiles due to its base version.  A test
# for a compatibility symbol which was superseded with a GLIBC_2.15
# version could be tested like this:
#
# ifdef HAVE-GLIBC_2.14
# tests += tst-spawn4-compat
# endif # HAVE-GLIBC_2.14
#
# (NB: GLIBC_2.14 is the symbol version that immediately precedes
# GLIBC_2.15.)

NF == 1 && $1 != "}" {
  haveversion[$1] = 1
}
END {
  for (i in haveversion)
    printf "have-%s = yes\n", i
}
