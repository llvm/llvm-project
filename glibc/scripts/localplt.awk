# This awk script expects to get command-line files that are each
# the output of 'readelf -WSdr' on a single shared object, and named
# .../NAME.jmprel where NAME is the unadorned file name of the shared object.
# It writes "NAME: SYMBOL" for each PLT entry in NAME that refers to a
# symbol defined in the same object.

BEGIN { result = 0 }

FILENAME != lastfile {
  if (lastfile && jmprel_offset == 0 && rela_offset == 0 && rel_offset == 0) {
    print FILENAME ": *** failed to find expected output (readelf -WSdr)";
    result = 2;
  }
  lastfile = FILENAME;
  jmprel_offset = 0;
  rela_offset = 0;
  rel_offset = 0;
  delete section_offset_by_address;
}

/^Section Headers:/ { in_shdrs = 1; next }
in_shdrs && !/^ +\[/ { in_shdrs = 0 }

in_shdrs && /^ +\[/ { sub(/\[ +/, "[") }
in_shdrs {
  address = strtonum("0x" $4);
  offset = strtonum("0x" $5);
  section_offset_by_address[address] = offset;
}

in_shdrs { next }

$1 == "Offset" && $2 == "Info" { in_relocs = 1; next }
NF == 0 { in_relocs = 0 }

in_relocs && relocs_offset == jmprel_offset && NF >= 5 {
  # Relocations against GNU_IFUNC symbols are not shown as an hexadecimal
  # value, but rather as the resolver symbol followed by ().
  if ($4 ~ /\(\)/) {
    print whatfile, gensub(/@.*/, "", "g", $5)
  } else {
    symval = strtonum("0x" $4);
    if (symval != 0)
      print whatfile, gensub(/@.*/, "", "g", $5)
  }
}

in_relocs && relocs_offset == rela_offset && NF >= 5 {
  # Relocations against GNU_IFUNC symbols are not shown as an hexadecimal
  # value, but rather as the resolver symbol followed by ().
  if ($4 ~ /\(\)/) {
    print whatfile, gensub(/@.*/, "", "g", $5), "RELA", $3
  } else {
    symval = strtonum("0x" $4);
    if (symval != 0)
      print whatfile, gensub(/@.*/, "", "g", $5), "RELA", $3
  }
}

in_relocs && relocs_offset == rel_offset && NF >= 5 {
  # Relocations against GNU_IFUNC symbols are not shown as an hexadecimal
  # value, but rather as the resolver symbol followed by ().
  if ($4 ~ /\(\)/) {
    print whatfile, gensub(/@.*/, "", "g", $5), "REL", $3
  } else {
    symval = strtonum("0x" $4);
    if (symval != 0)
      print whatfile, gensub(/@.*/, "", "g", $5), "REL", $3
  }
}

in_relocs { next }

$1 == "Relocation" && $2 == "section" && $5 == "offset" {
  relocs_offset = strtonum($6);
  whatfile = gensub(/^.*\/([^/]+)\.jmprel$/, "\\1:", 1, FILENAME);
  next
}

$2 == "(JMPREL)" {
  jmprel_addr = strtonum($3);
  if (jmprel_addr in section_offset_by_address) {
    jmprel_offset = section_offset_by_address[jmprel_addr];
  } else {
    print FILENAME ": *** DT_JMPREL does not match any section's address";
    result = 2;
  }
  next
}

$2 == "(RELA)" {
  rela_addr = strtonum($3);
  if (rela_addr in section_offset_by_address) {
    rela_offset = section_offset_by_address[rela_addr];
  } else {
    print FILENAME ": *** DT_RELA does not match any section's address";
    result = 2;
  }
  next
}

$2 == "(REL)" {
  rel_addr = strtonum($3);
  if (rel_addr in section_offset_by_address) {
    rel_offset = section_offset_by_address[rel_addr];
  } else {
    print FILENAME ": *** DT_REL does not match any section's address";
    result = 2;
  }
  next
}
END { exit(result) }
