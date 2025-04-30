# awk script for soversions.i -> gnu/lib-names.h; see Makeconfig.

#
{
  split($1, fields, "=")
  lib = fields[1];
  soname = version = fields[2];
  sub(/^.*=/, "", soname);
  sub(/^lib.*\.so\./, "", version);
  if ($soname !~ /^lib/) {
    extra = soname;
    sub(/\.so.*$/, "", extra);
  }
  else {
    extra = "";
  }
  soname = "\"" soname "\"";
  lib = toupper(lib);
  extra = toupper(extra);
  gsub(/-/, "_", lib);
  gsub(/-/, "_", extra);
  macros[$1 FS lib "_SO"] = soname;
  if (extra)
    macros[$1 FS extra "_SO"] = soname;
}

END {
  for (elt in macros) {
    split(elt, x);
    printf("%-40s%s\n", "#define " x[2], macros[elt]);
  }
}
