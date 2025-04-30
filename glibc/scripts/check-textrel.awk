# This awk script expects to get command-line files that are each
# the output of 'readelf -d' on a single shared object.
# It exits successfully (0) if none contained any TEXTREL markers.
# It fails (1) if any did contain a TEXTREL marker.
# It fails (2) if the input did not take the expected form.

BEGIN { result = textrel = sanity = 0 }

function check_one(name) {
  if (!sanity) {
    print name ": *** input did not look like readelf -d output";
    result = 2;
  } else if (textrel) {
    print name ": *** text relocations used";
    result = result ? result : 1;
  } else {
    print name ": OK";
  }

  textrel = sanity = 0;
}

FILENAME != lastfile {
  if (lastfile)
    check_one(lastfile);
  lastfile = FILENAME;
}

$1 == "Tag" && $2 == "Type" { sanity = 1 }
$2 == "(TEXTREL)" { textrel = 1 }
$2 == "(FLAGS)" {
  for (i = 3; i <= NF; ++i) {
    if ($i == "TEXTREL")
      textrel = 1;
  }
}

END {
  check_one(lastfile);
  exit(result);
}
