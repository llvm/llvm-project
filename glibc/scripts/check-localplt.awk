# This is an awk script to process the output of elf/check-localplt.
# The first file argument is the file of expected results.
# Each line is either a comment starting with # or it looks like:
#	libfoo.so: function
# or
#	libfoo.so: function + {RELA|REL} RELOC
# or
#	libfoo.so: function ?
# The first entry means that one is required.
# The second entry means that one is required and relocation may also be
# {RELA|REL} RELOC.
# The third entry means that a PLT entry for function is optional in
# libfoo.so.
# The second file argument is - and this (stdin) receives the output
# of the check-localplt program.

BEGIN { result = 0 }

FILENAME != "-" && /^#/ { next }

FILENAME != "-" {
  if (NF == 5 && $3 == "+" && ($4 == "RELA" || $4 == "REL")) {
    accept_type[$1 " " $2] = $4;
    accept_reloc[$1 " " $2] = $5;
  } else if (NF != 2 && !(NF == 3 && $3 == "?")) {
    printf "%s:%d: bad data line: %s\n", FILENAME, FNR, $0 > "/dev/stderr";
    result = 2;
  } else {
    accept[$1 " " $2] = NF == 2;
  }
  next;
}

NF != 2 && !(NF == 4 && ($3 == "RELA" || $3 == "REL")) {
  print "Unexpected output from check-localplt:", $0 > "/dev/stderr";
  result = 2;
  next
}

{
  key = $1 " " $2
  if ($3 == "RELA" || $3 == "REL") {
    # Entries like:
    # libc.so: free + RELA R_X86_64_GLOB_DAT
    # may be ignored.
    if (key in accept_type && accept_type[key] == $3 && accept_reloc[key] == $4) {
      # Match
      # libc.so: free + RELA R_X86_64_GLOB_DAT
      delete accept_type[key]
    }
  } else if (NF == 2 && key in accept_reloc) {
    # Match
    # libc.so: free
    # against
    # libc.so: free + RELA R_X86_64_GLOB_DAT
    if (key in accept_type)
      delete accept_type[key]
  } else if (key in accept) {
    delete accept[key]
  } else {
    print "Extra PLT reference:", $0;
    if (result == 0)
      result = 1;
  }
}

END {
  for (key in accept) {
    if (accept[key]) {
      # It's mandatory.
      print "Missing required PLT reference:", key;
      result = 1;
    }
  }

  for (key in accept_type) {
    # It's mandatory.
    print "Missing required PLT or " accept_reloc[key] " reference:", key;
    result = 1;
  }

  exit(result);
}
