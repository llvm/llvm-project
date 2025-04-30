BEGIN {
  print "#include <errno.h>";
  print "#include <mach/error.h>";
  print "#include <errorlib.h>";
  print "#define static static const";
  nsubs = split(subsys, subs);
  while (nsubs > 0) printf "#include \"%s\"\n", subs[nsubs--];
  print "\n\n\
const struct error_system __mach_error_systems[err_max_system + 1] =";
  print "  {";
}
/^static.*err_[a-z0-9A-Z_]+_sub *\[/ {
  s = $0; sub(/^.*err_/, "", s); sub(/_sub.*$/, "", s);
  printf "    [err_get_system (err_%s)] = { errlib_count (err_%s_sub),",
	s, s;
  printf "\"(system %s) error with unknown subsystem\", err_%s_sub },\n",
	s, s;
}
END {
  print "  };";
  print "\n\
const int __mach_error_system_count = errlib_count (__mach_error_systems);";
}
