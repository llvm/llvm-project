# This awk script expects to get command-line files that are each
# the output of 'readelf -l' on a single shared object.
# But the first file should contain just "execstack-no" or "execstack-yes",
# indicating what the default is in the absence of PT_GNU_STACK.
# It exits successfully (0) if none indicated executable stack.
# It fails (1) if any did indicate executable stack.
# It fails (2) if the input did not take the expected form.

BEGIN {
  result = sanity = 0; default_exec = -1;
  split(xfail, xfails, " ");
  for (x in xfails)
    expected_fails[xfails[x] ".phdr"] = 1;
}

/^execstack-no$/ { default_exec = 0; next }
/^execstack-yes$/ { default_exec = 1; next }

function check_one(name) {
  if (default_exec == -1) {
    print "*** missing execstack-default file?";
    result = 2;
  }

  n = split(name, parts, "/");
  basename = parts[n];
  expected_fail = basename in expected_fails;

  if (!sanity) {
    print name ": *** input did not look like readelf -l output";
    result = 2;
  } else if (stack_line) {
    if (stack_line ~ /^.*RW .*$/) {
      print name ": OK";
    } else if (stack_line ~ /^.*E.*$/) {
      if (expected_fail) {
	print name ": *** executable stack signaled, expected";
      } else {
	print name ": *** executable stack signaled";
	result = result ? result : 1;
      }
    }
  } else if (default_exec) {
    if (expected_fail) {
      print name ": *** no PT_GNU_STACK entry, expected";
    } else {
      print name ": *** no PT_GNU_STACK entry";
      result = result ? result : 1;
    }
  } else {
    print name ": no PT_GNU_STACK but default is OK";
  }

  sanity = 0;
}

FILENAME != lastfile {
  if (lastfile)
    check_one(lastfile);
  lastfile = FILENAME;
}

$1 == "Type" && $7 == "Flg" { sanity = 1; stack_line = "" }
$1 == "GNU_STACK" { stack_line = $0 }

END {
  check_one(lastfile);
  exit(result);
}
