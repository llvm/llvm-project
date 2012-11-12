#!/usr/bin/perl -w
#
# Simple little Perl script that takes the cxx-sections.data file as
# input and generates a directory structure that mimics the standard's
# structure.
use English;

$current_indent_level = -4;
while ($line = <STDIN>) {
  $line =~ /^\s*/;
  $next_indent_level = length($MATCH);
  if ($line =~ /\[([^\]]*)\]/) {
    my $section = $1;
    while ($next_indent_level < $current_indent_level) {
      chdir("..");
      $current_indent_level -= 4;
    }

    if ($next_indent_level == $current_indent_level) {
      chdir("..");
    } else {
      $current_indent_level = $next_indent_level;
    }
    mkdir($section);
    chdir($section);
  }
}
