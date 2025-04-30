BEGIN { calls="" }

{
  calls = calls " " $1;
  print "sysno-" $1 " = " $2;
  print "nargs-" $1 " = " $3;
}

END { print "mach-syscalls := " calls }
