#! @PERL@
eval "exec @PERL@ -S $0 $@"
    if 0;
# Copyright (C) 1997-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
# Contributed by Ulrich Drepper <drepper@gnu.org>, 1997.
# Based on the mtrace.awk script.

# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

$VERSION = "@VERSION@";
$PKGVERSION = "@PKGVERSION@";
$REPORT_BUGS_TO = '@REPORT_BUGS_TO@';
$progname = $0;

sub usage {
    print "Usage: mtrace [OPTION]... [Binary] MtraceData\n";
    print "  --help       print this help, then exit\n";
    print "  --version    print version number, then exit\n";
    print "\n";
    print "For bug reporting instructions, please see:\n";
    print "$REPORT_BUGS_TO.\n";
    exit 0;
}

# We expect two arguments:
#   #1: the complete path to the binary
#   #2: the mtrace data filename
# The usual options are also recognized.

arglist: while (@ARGV) {
    if ($ARGV[0] eq "--v" || $ARGV[0] eq "--ve" || $ARGV[0] eq "--ver" ||
	$ARGV[0] eq "--vers" || $ARGV[0] eq "--versi" ||
	$ARGV[0] eq "--versio" || $ARGV[0] eq "--version") {
	print "mtrace $PKGVERSION$VERSION\n";
	print "Copyright (C) 2021 Free Software Foundation, Inc.\n";
	print "This is free software; see the source for copying conditions.  There is NO\n";
	print "warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n";
	print "Written by Ulrich Drepper <drepper\@gnu.org>\n";

	exit 0;
    } elsif ($ARGV[0] eq "--h" || $ARGV[0] eq "--he" || $ARGV[0] eq "--hel" ||
	     $ARGV[0] eq "--help") {
	&usage;
    } elsif ($ARGV[0] =~ /^-/) {
	print "$progname: unrecognized option `$ARGV[0]'\n";
	print "Try `$progname --help' for more information.\n";
	exit 1;
    } else {
	last arglist;
    }
}

if ($#ARGV == 0) {
    $binary="";
    $data=$ARGV[0];
} elsif ($#ARGV == 1) {
    $binary=$ARGV[0];
    $data=$ARGV[1];

    if ($binary =~ /^.*[\/].*$/) {
	$prog = $binary;
    } else {
	$prog = "./$binary";
    }
    if (open (LOCS, "env LD_TRACE_LOADED_OBJECTS=1 $prog |")) {
	while (<LOCS>) {
	    chop;
	    if (/^.*=> (.*) .(0x[0123456789abcdef]*).$/) {
		$locs{$1} = $2;
	    }
	}
	close (LOCS);
    }
} else {
    die "Wrong number of arguments, run $progname --help for help.";
}

sub location {
    my $str = pop(@_);
    return $str if ($str eq "");
    if ($str =~ /.*[[](0x[^]]*)]:(.)*/) {
	my $addr = $1;
	my $fct = $2;
	return $cache{$addr} if (exists $cache{$addr});
	if ($binary ne "" && open (ADDR, "addr2line -e $binary $addr|")) {
	    my $line = <ADDR>;
	    chomp $line;
	    close (ADDR);
	    if ($line ne '??:0') {
		$cache{$addr} = $line;
		return $cache{$addr};
	    }
	}
	$cache{$addr} = $str = "$fct @ $addr";
    } elsif ($str =~ /^(.*):.*[[](0x[^]]*)]$/) {
	my $prog = $1;
	my $addr = $2;
	my $searchaddr;
	return $cache{$addr} if (exists $cache{$addr});
	if ($locs{$prog} ne "") {
	    $searchaddr = sprintf "%#x", $addr - $locs{$prog};
	} else {
	    $searchaddr = $addr;
	    $prog = $binary;
	}
	if ($binary ne "" && open (ADDR, "addr2line -e $prog $searchaddr|")) {
	    my $line = <ADDR>;
	    chomp $line;
	    close (ADDR);
	    if ($line ne '??:0') {
		$cache{$addr} = $line;
		return $cache{$addr};
	    }
	}
	$cache{$addr} = $str = $addr;
    } elsif ($str =~ /^.*[[](0x[^]]*)]$/) {
	my $addr = $1;
	return $cache{$addr} if (exists $cache{$addr});
	if ($binary ne "" && open (ADDR, "addr2line -e $binary $addr|")) {
	    my $line = <ADDR>;
	    chomp $line;
	    close (ADDR);
	    if ($line ne '??:0') {
		$cache{$addr} = $line;
		return $cache{$addr};
	    }
	}
	$cache{$addr} = $str = $addr;
    }
    return $str;
}

$nr=0;
open(DATA, "<$data") || die "Cannot open mtrace data file";
while (<DATA>) {
    my @cols = split (' ');
    my $n, $where;
    if ($cols[0] eq "@") {
	# We have address and/or function name.
	$where=$cols[1];
	$n=2;
    } else {
	$where="";
	$n=0;
    }

    $allocaddr=$cols[$n + 1];
    $howmuch=hex($cols[$n + 2]);

    ++$nr;
    SWITCH: {
	if ($cols[$n] eq "+") {
	    if (defined $allocated{$allocaddr}) {
		printf ("+ %#0@XXX@x Alloc %d duplicate: %s %s\n",
			hex($allocaddr), $nr, &location($addrwas{$allocaddr}),
			$where);
	    } elsif ($allocaddr =~ /^0x/) {
		$allocated{$allocaddr}=$howmuch;
		$addrwas{$allocaddr}=$where;
	    }
	    last SWITCH;
	}
	if ($cols[$n] eq "-") {
	    if (defined $allocated{$allocaddr}) {
		undef $allocated{$allocaddr};
		undef $addrwas{$allocaddr};
	    } else {
		printf ("- %#0@XXX@x Free %d was never alloc'd %s\n",
			hex($allocaddr), $nr, &location($where));
	    }
	    last SWITCH;
	}
	if ($cols[$n] eq "<") {
	    if (defined $allocated{$allocaddr}) {
		undef $allocated{$allocaddr};
		undef $addrwas{$allocaddr};
	    } else {
		printf ("- %#0@XXX@x Realloc %d was never alloc'd %s\n",
			hex($allocaddr), $nr, &location($where));
	    }
	    last SWITCH;
	}
	if ($cols[$n] eq ">") {
	    if (defined $allocated{$allocaddr}) {
		printf ("+ %#0@XXX@x Realloc %d duplicate: %#010x %s %s\n",
			hex($allocaddr), $nr, $allocated{$allocaddr},
			&location($addrwas{$allocaddr}), &location($where));
	    } else {
		$allocated{$allocaddr}=$howmuch;
		$addrwas{$allocaddr}=$where;
	    }
	    last SWITCH;
	}
	if ($cols[$n] eq "=") {
	    # Ignore "= Start".
	    last SWITCH;
	}
	if ($cols[$n] eq "!") {
	    # Ignore failed realloc for now.
	    last SWITCH;
	}
    }
}
close (DATA);

# Now print all remaining entries.
@addrs= keys %allocated;
$anything=0;
if ($#addrs >= 0) {
    foreach $addr (sort @addrs) {
	if (defined $allocated{$addr}) {
	    if ($anything == 0) {
		print "\nMemory not freed:\n-----------------\n";
		print ' ' x (@XXX@ - 7), "Address     Size     Caller\n";
		$anything=1;
	    }
	    printf ("%#0@XXX@x %#8x  at %s\n", hex($addr), $allocated{$addr},
		    &location($addrwas{$addr}));
	}
    }
}
print "No memory leaks.\n" if ($anything == 0);

exit $anything != 0;
