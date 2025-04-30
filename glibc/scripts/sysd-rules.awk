# This is a GAWK script to generate the sysd-rules file.
# It does not read any input, but it requires that several variables
# be set on its command line (using -v) to their makefile counterparts:
#	all_object_suffixes	$(all-object-suffixes)
#	inhibit_sysdep_asm	$(inhibit-sysdep-asm)
#	config_sysdirs		$(config_sysdirs)
#	sysd_rules_patterns	$(sysd-rules-patterns)

BEGIN {
  print "sysd-rules-sysdirs :=", config_sysdirs;

  nsuffixes = split(all_object_suffixes, suffixes);
  ninhibit_asm = split(inhibit_sysdep_asm, inhibit_asm);
  nsysdirs = split(config_sysdirs, sysdirs);
  npatterns = split(sysd_rules_patterns, patterns);

  # Each element of $(sysd-rules-patterns) is a pair TARGET:DEP.
  # They are no in particular order.  We need to sort them so that
  # the longest TARGET is first, and, among elements with the same
  # TARGET, the longest DEP is first.
  for (i = 1; i <= npatterns; ++i) {
    if (split(patterns[i], td, ":") != 2) {
      msg = "bad sysd-rules-patterns element '" patterns[i] "'";
      print msg > "/dev/stderr";
      exit 2;
    }
    target_order = sprintf("%09d", npatterns + 1 - length(td[1]));
    dep_order = sprintf("%09d", npatterns - length(td[2]));
    sort_patterns[target_order SUBSEP dep_order] = patterns[i];
  }
  asorti(sort_patterns, map_patterns);
  for (i in map_patterns) {
    patterns[i] = sort_patterns[map_patterns[i]];
  }

  for (sysdir_idx = 1; sysdir_idx <= nsysdirs; ++sysdir_idx) {
    dir = sysdirs[sysdir_idx];
    if (dir !~ /^\//) dir = "$(..)" dir;
    asm_rules = 1;
    for (i = 1; i <= ninhibit_asm; ++i) {
      if (dir ~ ("^.*sysdeps/" inhibit_asm[i] "$")) {
        asm_rules = 0;
        break;
      }
    }
    for (suffix_idx = 1; suffix_idx <= nsuffixes; ++suffix_idx) {
      o = suffixes[suffix_idx];
      for (pattern_idx = 1; pattern_idx <= npatterns; ++pattern_idx) {
        pattern = patterns[pattern_idx];
        split(pattern, td, ":");
        target_pattern = td[1];
        dep_pattern = td[2];
        # rtld objects are always PIC.
        if (target_pattern ~ /^rtld/ && o != ".os") {
            continue;
        }
        if (target_pattern == "%") {
          command_suffix = "";
        } else {
          prefix = gensub(/%/, "", 1, target_pattern);
          command_suffix = " $(" prefix  "CPPFLAGS)" " $(" prefix  "CFLAGS)";
        }
        target = "$(objpfx)" target_pattern o ":";
        if (asm_rules) {
          dep = dir "/" dep_pattern ".S";
          print target, dep, "$(before-compile)";
          print "\t$(compile-command.S)" command_suffix;
        }
        dep = dir "/" dep_pattern ".c";
        print target, dep, "$(before-compile)";
        print "\t$(compile-command.c)" command_suffix;
      }
    }
    print "$(inst_includedir)/%.h:", dir "/%.h", "$(+force)";
    print "\t$(do-install)";
  }

  print "sysd-rules-done := t";
  exit 0;
}
