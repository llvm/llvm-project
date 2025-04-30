# Script to preprocess Versions.all lists based on "earliest version"
# specifications in the shlib-versions file.

# Return -1, 0 or 1 according to whether v1 is less than, equal to or
# greater than v2 as a version string. Simplified from GNU Autoconf
# version; this one does not need to handle .0x fraction-style versions.
function vers_compare (v1, v2)
{
  while (length(v1) && length(v2)) {
    if (v1 ~ /^[0-9]/ && v2 ~ /^[0-9]/) {
      for (len1 = 1; substr(v1, len1 + 1) ~ /^[0-9]/; len1++) continue;
      for (len2 = 1; substr(v2, len2 + 1) ~ /^[0-9]/; len2++) continue;
      d1 = substr(v1, 1, len1); v1 = substr(v1, len1 + 1);
      d2 = substr(v2, 1, len2); v2 = substr(v2, len2 + 1);
      d1 += 0;
      d2 += 0;
    } else {
      d1 = substr(v1, 1, 1); v1 = substr(v1, 2);
      d2 = substr(v2, 1, 1); v2 = substr(v2, 2);
    }
    if (d1 < d2) return -1;
    if (d1 > d2) return 1;
  }
  if (length(v2)) return -1;
  if (length(v1)) return 1;
  return 0;
}

NF > 2 && $2 == ":" {
  for (i = 0; i <= NF - 3; ++i)
    firstversion[$1, i] = $(3 + i);
  idx[$1] = 0;
  next;
}

NF == 2 && $2 == "{" { thislib = $1; print; next }

$1 == "}" {
  if ((thislib, idx[thislib]) in firstversion) {
    # We haven't seen the stated version, but have produced
    # others pointing to it, so we synthesize it now.
    printf "  %s\n", firstversion[thislib, idx[thislib]];
    idx[thislib]++;
  }
  print;
  next;
}

/GLIBC_PRIVATE/ { print; next }

{
  if ((thislib, idx[thislib]) in firstversion) {
    f = v = firstversion[thislib, idx[thislib]];
    while (vers_compare($1, v) >= 0) {
      delete firstversion[thislib, idx[thislib]];
      idx[thislib]++;
      if ((thislib, idx[thislib]) in firstversion) {
        # If we're skipping a referenced version to jump ahead to a
        # later version, synthesize the earlier referenced version now.
        if (v != $1 && (thislib, v) in usedversion)
          print "  " v;
        v = firstversion[thislib, idx[thislib]];
      } else
        break;
    }
    if ($1 == v || $1 == f)
      # This version was the specified earliest version itself.
      print;
    else if (vers_compare($1, v) < 0) {
      # This version is older than the specified earliest version.
      print "  " $1, "=", v;
      # Record that V has been referred to, so we will be sure to emit it
      # if we hit a later one without hitting V itself.
      usedversion[thislib, v] = 1;
    }
    else {
      # This version is newer than the specified earliest version.
      # We haven't seen that version itself or else we wouldn't be here
      # because we would have removed it from the firstversion array.
      # If there were any earlier versions that used that one, emit it now.
      if ((thislib, v) in usedversion) {
        print "  " v;
      }
      print "  " $1;
    }
  }
  else
    print;
}
