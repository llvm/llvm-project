#!/usr/bin/perl

#
#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//
#

use strict;
use warnings;

use FindBin;
use lib "$FindBin::Bin/lib";

use tools;

our $VERSION = "0.004";
my $target_os;
my $target_arch;
my $target_mic_arch;

my $hex = qr{[0-9a-f]}i;    # hex digit.

# mic-specific details.

sub bad_mic_fmt($) {
    # Before we allowed both elf64-x86-64-freebsd and elf-l1om-freebsd.
    # Now the first one is obsolete, only elf64-l1om-freebsd is allowed.
    my ( $fmt ) = @_;
    if ( 0 ) {
    } elsif ( "$target_mic_arch" eq "knf" ) {
	    return $fmt !~ m{\Aelf64-l1om?\z};
    } elsif ( "$target_mic_arch" eq "knc" ) {
	    return $fmt !~ m{\Aelf64-k1om?\z};
	} else {
	    return 1;
	};
}; # sub bad_mic_fmt

# Undesired instructions for mic: all x87 and some other.
# AC: Since compiler 2010-06-30 x87 instructions are supported, removed the check of x87.
my $mic_bad_re;
sub bad_mic_instr($$) {
    my ( $instr, $args ) = @_;
    if ( "$target_mic_arch" eq "knc" ) {
	# workaround of bad code generation on KNF Linux* OS:
	return ( defined( $instr ) and $instr =~ $mic_bad_re );
    } else {
	return ( defined( $instr ) and $instr =~ $mic_bad_re or defined( $args ) and $args =~ m{xmm}i );
    }
}; # sub bad_mic_instr

# lin_32-specific details.

sub bad_ia32_fmt($) {
    my ( $fmt ) = @_;
    return $fmt !~ m{\Aelf32-i386\z};
}; # sub bad_ia32_fmt

my @sse2 =
    qw{
        movapd movupd movhpd movlpd movmskpd movsd
        addpd addsd subpd subsd mulpd mulsd divpd divsd sqrtpd sqrtsd maxpd maxsd minpd minsd
        andpd andnpd orpd xorpd
        cmppd cmpsd comisd ucomisd
        shufpd unpckhpd unpcklpd
        cvtpd2pi cvttpd2pi cvtpi2pd cvtpd2dq cvttpd2dq cvtdq2pd cvtps2pd cvtpd2ps cvtss2sd cvtsd2ss
        cvtsd2si cvttsd2si cvtsi2sd cvtdq2ps cvtps2dq cvttps2dq movdqa movdqu movq2dq movdq2q
        pmuludq paddq psubq pshuflw pshufhw pshufd pslldq psrldq punpckhqdq punpcklqdq clflush
        lfence mfence maskmovdqu movntpd movntdq movnti
    };
my @sse3 =
    qw{
        fisttp lddqu addsubps addsubpd haddps hsubps haddpd hsubpd movshdup movsldup movddup monitor
        mwait
    };
my @ssse3 =
    qw{
        phaddw phaddsw phaddd phsubw phsubsw phsubd pabsb pabsw pabsd pmaddubsw pmulhrsw pshufb
        psignb psignw psignd palignr
    };
my @sse4 =
    (
        # SSE4.1
        qw{
            pmulld pmuldq dppd dpps movntdqa blendpd blendps blendvpd blendvps pblendvb pblendw pminuw
            pminud pminsb pminsd pmaxuw pmaxud pmaxsb pmaxsd roundps roundpd roundss roundsd extractps
            insertps pinsrb pinsrd pinsrq pextrb pextrw pextrd pextrq pmovsxbw pmovzxbw pmovsxbd
            pmovzxbd pmovsxwd pmovzxwd pmovsxbq pmovzxbq pmovsxwq pmovzxwq pmovsxdq pmovzxdq mpsadbw
            phminposuw ptest pcmpeqq packusdw
        },
        # SSE4.2
        qw{
            pcmpestri pcmpestrm pcmpistri pcmpistrm pcmpgtq crc32 popcnt
        }
    );

# Undesired instructions for IA-32 architecture: Pentium 4 (SSE2) and newer.
# TODO: It would be much more reliable to list *allowed* instructions rather than list undesired
# instructions. In such a case the list will be stable and not require update when SSE5 is released.
my @ia32_bad_list = ( @sse2, @sse3, @ssse3, @sse4 );

my $ia32_bad_re = qr{@{[ "^(?:" . join( "|", @ia32_bad_list ) . ")" ]}}i;

sub bad_ia32_instr($$) {
    my ( $instr, $args ) = @_;
    return ( defined( $instr ) and $instr =~ $ia32_bad_re );
}; # sub bad_ia32_instr

sub check_file($;$$) {

    my ( $file, $show_instructions, $max_instructions ) = @_;
    my @bulk;

    if ( not defined( $max_instructions ) ) {
        $max_instructions = 100;
    }; # if

    execute( [ "x86_64-k1om-linux-objdump", "-d", $file ], -stdout => \@bulk );

    my $n = 0;
    my $errors = 0;
    my $current_func  = "";    # Name of current function.
    my $reported_func = "";    # name of last reported function.
    foreach my $line ( @bulk ) {
        ++ $n;
        if ( 0 ) {
        } elsif ( $line =~ m{^\s*$} ) {
            # Empty line.
            # Ignore.
        } elsif ( $line =~ m{^In archive (.*?):\s*$} ) {
            # In archive libomp.a:
        } elsif ( $line =~ m{^(?:.*?):\s*file format (.*?)\s*$} ) {
            # libomp.so:     file format elf64-x86-64-freebsd
            # kmp_ftn_cdecl.o:     file format elf64-x86-64
            my $fmt = $1;
            if ( bad_fmt( $fmt ) ) {
                runtime_error( "Invalid file format: $fmt." );
            }; # if
        } elsif ( $line =~ m{^Disassembly of section (.*?):\s*$} ) {
            # Disassembly of section .plt:
        } elsif ( $line =~ m{^$hex+ <([^>]+)>:\s*$} ) {
            # 0000000000017e98 <__kmp_str_format@plt-0x10>:
            $current_func = $1;
        } elsif ( $line =~ m{^\s*\.{3}\s*$} ) {
        } elsif ( $line =~ m{^\s*($hex+):\s+($hex$hex(?: $hex$hex)*)\s+(?:lock\s+|rex[.a-z]*\s+)?([^ ]+)(?:\s+([^#]+?))?\s*(?:#|$)} ) {
            #   17e98:       ff 35 fa 7d 26 00       pushq  0x267dfa(%rip)        # 27fc98 <_GLOBAL_OFFSET_TABLE>
            my ( $addr, $dump, $instr, $args ) = ( $1, $2, $3, $4 );
            # Check this is not a bad instruction and xmm registers are not used.
            if ( bad_instr( $instr, $args ) ) {
                if ( $errors == 0 ) {
                    warning( "Invalid instructions found in `$file':" );
                }; # if
                if ( $current_func ne $reported_func ) {
                    warning( "    $current_func" );
                    $reported_func = $current_func;
                }; # if
                ++ $errors;
                if ( $show_instructions ) {
                    warning( "        $line" );
                }; # if
                if ( $errors >= $max_instructions ) {
                    info( "$errors invalid instructions found; scanning stopped." );
                    last;
                }; # if
            }; # if
        } else {
            runtime_error( "Error parsing objdump output line $n:\n>>>> $line\n" );
        }; # if
    }; # foreach $line

    return $errors;

}; # sub check_file

# --------------------------------------------------------------------------------------------------

# Parse command line.
my $max_instructions;
my $show_instructions;
get_options(
    "os=s"               => \$target_os,
    "arch=s"             => \$target_arch,
    "mic-arch=s"         => \$target_mic_arch,
    "max-instructions=i" => \$max_instructions,
    "show-instructions!" => \$show_instructions,
);
my $target_platform = $target_os . "_" . $target_arch;
if ( "$target_os" eq "lin" and "$target_mic_arch" eq "knf" ) {
    $mic_bad_re = qr{^(?:pause|[slm]fence|scatter|gather|cmpxchg16b|clevict[12])}i;
} else {
    $mic_bad_re = qr{^(?:pause|[slm]fence|scatter|gather|cmov|cmpxchg16b|clevict[12])}i;
};
if ( 0 ) {
} elsif ( $target_platform eq "lin_mic" ) {
    *bad_instr = \*bad_mic_instr;
    *bad_fmt   = \*bad_mic_fmt;
} elsif ( $target_platform eq "lin_32" ) {
    *bad_instr = \*bad_ia32_instr;
    *bad_fmt   = \*bad_ia32_fmt;
} else {
    runtime_error( "Only works on lin_32 and lin_mic platforms." );
}; # if

# Do the work.
my $rc = 0;
if ( not @ARGV ) {
    info( "No arguments specified -- nothing to do." );
} else {
    foreach my $arg ( @ARGV ) {
        my $errs = check_file( $arg, $show_instructions, $max_instructions );
        if ( $errs > 0 ) {
            $rc = 3;
        }; # if
    }; # foreach $arg
}; # if

exit( $rc );

__END__

=pod

=head1 NAME

B<check-instruction-set.pl> -- Make sure binary file does not contain undesired instructions.

=head1 SYNOPSIS

B<check-instructions.pl> I<option>... I<file>...

=head1 OPTIONS

=over

=item B<--architecture=>I<arch>

Specify target architecture.

=item B<--max-instructions=>I<number>

Stop scanning if I<number> invalid instructions found. 100 by default.

=item B<--os=>I<os>

Specify target OS.

=item B<-->[B<no->]B<show-instructions>

Show invalid instructions found in the file. Bu default, instructions are not shown.

=item Standard Options

=over

=item B<--doc>

=item B<--manual>

Print full help message and exit.

=item B<--help>

Print short help message and exit.

=item B<--usage>

Print very short usage message and exit.

=item B<--verbose>

Do print informational messages.

=item B<--version>

Print program version and exit.

=item B<--quiet>

Work quiet, do not print informational messages.

=back

=back

=head1 ARGUMENTS

=over

=item I<file>

File (object file or library, either static or dynamic) to check.

=back

=head1 DESCRIPTION

The script runs F<objdump> utility to get disassembler listing and checks the file does not contain
unwanted instructions.

Currently the script works only for:

=over

=item C<lin_mic>

Intel(R) Many Integrated Core Architecture target OS. Undesired unstructions are: all x87 instructions and some others.

=item C<lin_32>

Undesired instructions are instructions not valid for Pentium 3 processor (SSE2 and newer).

=back

=cut

