#!/usr/bin/env perl
# Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

package hipvars;
use warnings;
use Getopt::Long;
use Cwd;
use File::Basename;

$HIP_BASE_VERSION_MAJOR = "6";
$HIP_BASE_VERSION_MINOR = "2";
$HIP_BASE_VERSION_PATCH = "0";

#---
# Function to parse config file
sub parse_config_file {
    my ($file, $config) = @_;
    if (open (CONFIG, "$file")) {
        while (<CONFIG>) {
            my $config_line=$_;
            chop ($config_line);
            $config_line =~ s/^\s*//;
            $config_line =~ s/\s*$//;
            if (($config_line !~ /^#/) && ($config_line ne "")) {
                my ($name, $value) = split (/=/, $config_line);
                $$config{$name} = $value;
            }
        }
        close(CONFIG);
    }
}

#---
# Function to check if executable can be run
sub can_run {
    my ($exe) = @_;
    `$exe --version 2>&1`;
    if ($? == 0) {
        return 1;
    } else {
        return 0;
    }
}

$isWindows =  ($^O eq 'MSWin32' or $^O eq 'msys');
$doubleQuote = "\"";

#
# TODO: Fix rpath LDFLAGS settings
#
# Since this hipcc script gets installed at two uneven hierarchical levels,
# linked by symlink, the absolute path of this script should be used to
# derive HIP_PATH, as dirname $0 could be /opt/rocm/bin or /opt/rocm/hip/bin
# depending on how it gets invoked.
# ROCM_PATH which points to <rocm_install_dir> is determined based on whether
# we find bin/rocm_agent_enumerator in the parent of HIP_PATH or not. If it is found,
# ROCM_PATH is defined relative to HIP_PATH else it is hardcoded to /opt/rocm.
#
$HIP_PATH=$ENV{'HIP_PATH'} // dirname(Cwd::abs_path("$0/../")); # use parent directory of hipcc
if ($isWindows and defined $ENV{'HIP_PATH'}) {
  $HIP_PATH =~ s/^"(.*)"$/$1/;
  $HIP_PATH =~ s/\\/\//g;
}
if (-e "$HIP_PATH/bin/rocm_agent_enumerator") {
    $ROCM_PATH=$ENV{'ROCM_PATH'} // "$HIP_PATH"; # use HIP_PATH
}elsif (-e "$HIP_PATH/../bin/rocm_agent_enumerator") { # case for backward compatibility
    $ROCM_PATH=$ENV{'ROCM_PATH'} // dirname("$HIP_PATH"); # use parent directory of HIP_PATH
} else {
    $ROCM_PATH=$ENV{'ROCM_PATH'} // "/opt/rocm";
}
$CUDA_PATH=$ENV{'CUDA_PATH'} // '/usr/local/cuda';
if ($isWindows and defined $ENV{'CUDA_PATH'}) {
  $CUDA_PATH =~ s/^"(.*)"$/$1/;
  $CUDA_PATH =~ s/\\/\//g;
}

# Windows/Distro's have a different structure, all binaries are with hipcc
if ($isWindows or -e "$HIP_PATH/bin/clang") {
    $HIP_CLANG_PATH=$ENV{'HIP_CLANG_PATH'} // "$HIP_PATH/bin";
} else {
    $HIP_CLANG_PATH=$ENV{'HIP_CLANG_PATH'} // "$ROCM_PATH/lib/llvm/bin";
}
# HIP_ROCCLR_HOME is used by Windows builds
$HIP_ROCCLR_HOME=$ENV{'HIP_ROCCLR_HOME'};

if (defined $HIP_ROCCLR_HOME) {
    $HIP_INFO_PATH= "$HIP_ROCCLR_HOME/lib/.hipInfo";
} else {
    $HIP_INFO_PATH= "$HIP_PATH/lib/.hipInfo"; # use actual file
}
#---
#HIP_PLATFORM controls whether to use nvidia or amd platform:
$HIP_PLATFORM=$ENV{'HIP_PLATFORM'};
# Read .hipInfo
my %hipInfo = ();
parse_config_file("$HIP_INFO_PATH", \%hipInfo);
# Prioritize Env first, otherwise use the hipInfo config file
$HIP_COMPILER = $ENV{'HIP_COMPILER'} // $hipInfo{'HIP_COMPILER'} // "clang";
$HIP_RUNTIME = $ENV{'HIP_RUNTIME'} // $hipInfo{'HIP_RUNTIME'} // "rocclr";

# If using ROCclr runtime, need to find HIP_ROCCLR_HOME
if (defined $HIP_RUNTIME and $HIP_RUNTIME eq "rocclr" and !defined $HIP_ROCCLR_HOME) {
    my $hipvars_dir = dirname(Cwd::abs_path($0));
    if (-e "$hipvars_dir/../lib/bitcode") {
        $HIP_ROCCLR_HOME = Cwd::abs_path($hipvars_dir . "/.."); #FILE_REORG Backward compatibility
    } elsif (-e "$hipvars_dir/lib/bitcode") {
        $HIP_ROCCLR_HOME = Cwd::abs_path($hipvars_dir);
    } else {
        $HIP_ROCCLR_HOME = $HIP_PATH; # use HIP_PATH
    }
}

if (not defined $HIP_PLATFORM) {
    if (can_run($doubleQuote . "$HIP_CLANG_PATH/clang++" . $doubleQuote) or can_run("amdclang++")) {
        $HIP_PLATFORM = "amd";
    } elsif (can_run($doubleQuote . "$CUDA_PATH/bin/nvcc" . $doubleQuote) or can_run("nvcc")) {
        $HIP_PLATFORM = "nvidia";
        $HIP_COMPILER = "nvcc";
        $HIP_RUNTIME = "cuda";
    } else {
        # Default to amd for now
        $HIP_PLATFORM = "amd";
    }
} elsif ($HIP_PLATFORM eq "hcc") {
    $HIP_PLATFORM = "amd";
    warn("Warning: HIP_PLATFORM=hcc is deprecated. Please use HIP_PLATFORM=amd. \n")
} elsif ($HIP_PLATFORM eq "nvcc") {
    $HIP_PLATFORM = "nvidia";
    $HIP_COMPILER = "nvcc";
    $HIP_RUNTIME = "cuda";
    warn("Warning: HIP_PLATFORM=nvcc is deprecated. Please use HIP_PLATFORM=nvidia. \n")
}

if ($HIP_COMPILER eq "clang") {
    # Windows does not have clang at linux default path
    if (defined $HIP_ROCCLR_HOME and (-e "$HIP_ROCCLR_HOME/bin/clang" or -e "$HIP_ROCCLR_HOME/bin/clang.exe")) {
        $HIP_CLANG_PATH = "$HIP_ROCCLR_HOME/bin";
    }
}

#---
# Read .hipVersion
my %hipVersion = ();
if ($isWindows) {
    parse_config_file("$hipvars::HIP_PATH/bin/.hipVersion", \%hipVersion);
} else {
    parse_config_file("$hipvars::HIP_PATH/share/hip/version", \%hipVersion);
}
$HIP_VERSION_MAJOR = $hipVersion{'HIP_VERSION_MAJOR'} // $HIP_BASE_VERSION_MAJOR;
$HIP_VERSION_MINOR = $hipVersion{'HIP_VERSION_MINOR'} // $HIP_BASE_VERSION_MINOR;
$HIP_VERSION_PATCH = $hipVersion{'HIP_VERSION_PATCH'} // $HIP_BASE_VERSION_PATCH;
$HIP_VERSION_GITHASH = $hipVersion{'HIP_VERSION_GITHASH'} // 0;
$HIP_VERSION="$HIP_VERSION_MAJOR.$HIP_VERSION_MINOR.$HIP_VERSION_PATCH-$HIP_VERSION_GITHASH";
