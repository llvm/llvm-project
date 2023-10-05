#!/usr/bin/env perl
# Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.
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

# Need perl > 5.10 to use logic-defined or
use 5.006; use v5.10.1;
use warnings;
use File::Basename;
use File::Temp qw/ :mktemp  /;
use Cwd;
use Cwd 'abs_path';

# HIP compiler driver
# Will call clang or nvcc (depending on target) and pass the appropriate include and library options for
# the target compiler and HIP infrastructure.

# Will pass-through options to the target compiler.  The tools calling HIPCC must ensure the compiler
# options are appropriate for the target compiler.

# Environment variable HIP_PLATFORM is to detect amd/nvidia path:
# HIP_PLATFORM='nvidia' or HIP_PLATFORM='amd'.
# If HIP_PLATFORM is not set hipcc will attempt auto-detect based on if nvcc is found.
#
# Other environment variable controls:
# HIP_PATH       : Path to HIP directory, default is one dir level above location of this script.
# CUDA_PATH      : Path to CUDA SDK (default /usr/local/cuda). Used on NVIDIA platforms only.
# HIP_ROCCLR_HOME : Path to HIP/ROCclr directory. Used on AMD platforms only.
# HIP_CLANG_PATH : Path to HIP-Clang (default to ../../llvm/bin relative to this
#                  script's abs_path). Used on AMD platforms only.

if(scalar @ARGV == 0){
    print "No Arguments passed, exiting ...\n";
    exit(-1);
}

# retrieve --rocm-path hipcc option from command line.
# We need to respect this over the env var ROCM_PATH for this compilation.
sub get_path_options {
  my $rocm_path="";
  my $hip_path="";
  my @CLArgs = @ARGV;
  foreach $arg (@CLArgs) {
    if (index($arg,"--rocm-path=") != -1) {
      ($rocm_path) = $arg=~ /=\s*(.*)\s*$/;
      next;
    }
    if (index($arg,"--hip-path=") != -1) {
      ($hip_path) = $arg=~ /=\s*(.*)\s*$/;
      next;
    }
  }
  return ($rocm_path, $hip_path);
}

$verbose = $ENV{'HIPCC_VERBOSE'} // 0;
# Verbose: 0x1=commands, 0x2=paths, 0x4=hipcc args

$HIPCC_COMPILE_FLAGS_APPEND=$ENV{'HIPCC_COMPILE_FLAGS_APPEND'};
$HIPCC_LINK_FLAGS_APPEND=$ENV{'HIPCC_LINK_FLAGS_APPEND'};

# Known Features
@knownFeatures = ('sramecc-', 'sramecc+', 'xnack-', 'xnack+');

$HIP_LIB_PATH=$ENV{'HIP_LIB_PATH'};
$DEVICE_LIB_PATH=$ENV{'DEVICE_LIB_PATH'};
$HIP_CLANG_HCC_COMPAT_MODE=$ENV{'HIP_CLANG_HCC_COMPAT_MODE'}; # HCC compatibility mode
$HIP_COMPILE_CXX_AS_HIP=$ENV{'HIP_COMPILE_CXX_AS_HIP'} // "1";

#---
# Temporary directories
my @tmpDirs = ();

#---
# Create a new temporary directory and return it
sub get_temp_dir {
    my $tmpdir = mkdtemp("/tmp/hipccXXXXXXXX");
    push (@tmpDirs, $tmpdir);
    return $tmpdir;
}

#---
# Delete all created temporary directories
sub delete_temp_dirs {
    if (@tmpDirs) {
        system ('rm -rf ' . join (' ', @tmpDirs));
    }
    return 0;
}

my $base_dir;
BEGIN {
    $base_dir = dirname(Cwd::realpath(__FILE__) );
    my ($rocm_path, $hip_path) = get_path_options();
    if ($rocm_path ne '') {
      # --rocm-path takes precedence over ENV{ROCM_PATH}
      $ENV{ROCM_PATH}=$rocm_path;
    }
    if ($hip_path ne '') {
      # --rocm-path takes precedence over ENV{ROCM_PATH}
      $ENV{HIP_PATH}=$hip_path;
    }
}
use lib "$base_dir/";

use hipvars;
$isWindows      =   $hipvars::isWindows;
$doubleQuote    =   $hipvars::doubleQuote;
$HIP_RUNTIME    =   $hipvars::HIP_RUNTIME;
$HIP_PLATFORM   =   $hipvars::HIP_PLATFORM;
$HIP_COMPILER   =   $hipvars::HIP_COMPILER;
$HIP_CLANG_PATH =   $hipvars::HIP_CLANG_PATH;
$CUDA_PATH      =   $hipvars::CUDA_PATH;
$HIP_PATH       =   $hipvars::HIP_PATH;
$ROCM_PATH      =   $hipvars::ROCM_PATH;
$HIP_VERSION    =   $hipvars::HIP_VERSION;
$HIP_ROCCLR_HOME =   $hipvars::HIP_ROCCLR_HOME;

sub get_normalized_path {
  return $doubleQuote . $_[0] . $doubleQuote;
}

if ($HIP_PLATFORM eq "amd") {
  $HIP_INCLUDE_PATH = "$HIP_ROCCLR_HOME/include";
  if (!defined $HIP_LIB_PATH) {
    $HIP_LIB_PATH = "$HIP_ROCCLR_HOME/lib";
  }
}

if ($verbose & 0x2) {
    print ("HIP_PATH=$HIP_PATH\n");
    print ("HIP_PLATFORM=$HIP_PLATFORM\n");
    print ("HIP_COMPILER=$HIP_COMPILER\n");
    print ("HIP_RUNTIME=$HIP_RUNTIME\n");
}

# set if user explicitly requests -stdlib=libc++. (else we default to libstdc++ for better interop with g++):
$setStdLib = 0;  # TODO - set to 0

$default_amdgpu_target = 1;

if ($HIP_PLATFORM eq "amd") {
    $execExtension = "";
    if($isWindows) {
        $execExtension = ".exe";
    }
    $HIPCC=get_normalized_path("$HIP_CLANG_PATH/clang++" . $execExtension);

    # If $HIPCC clang++ is not compiled, use clang instead
    if ( ! -e $HIPCC ) {
        $HIPCC=get_normalized_path("$HIP_CLANG_PATH/clang" . $execExtension);
        $HIPLDFLAGS = "--driver-mode=g++";
    }
    # to avoid using dk linker or MSVC linker
    if($isWindows) {
        $HIPLDFLAGS .= " -fuse-ld=lld";
        $HIPLDFLAGS .= " --ld-path=" . get_normalized_path("$HIP_CLANG_PATH/lld-link.exe");
    }

    # get Clang RT Builtin path 
    $HIP_CLANG_RT_LIB = `$HIPCC --print-runtime-dir`;
    chomp($HIP_CLANG_RT_LIB);

    if (! defined $HIP_INCLUDE_PATH) {
        $HIP_INCLUDE_PATH = "$HIP_PATH/include";
    }
    if (! defined $HIP_LIB_PATH) {
        $HIP_LIB_PATH = "$HIP_PATH/lib";
    }
    if ($verbose & 0x2) {
        print ("ROCM_PATH=$ROCM_PATH\n");
        if (defined $HIP_ROCCLR_HOME) {
            print ("HIP_ROCCLR_HOME=$HIP_ROCCLR_HOME\n");
        }
        print ("HIP_CLANG_PATH=$HIP_CLANG_PATH\n");
        print ("HIP_INCLUDE_PATH=$HIP_INCLUDE_PATH\n");
        print ("HIP_LIB_PATH=$HIP_LIB_PATH\n");
        print ("DEVICE_LIB_PATH=$DEVICE_LIB_PATH\n");
        print ("HIP_CLANG_RT_LIB=$HIP_CLANG_RT_LIB\n");
    }

    if ($HIP_CLANG_HCC_COMPAT_MODE) {
        ## Allow __fp16 as function parameter and return type.
        $HIPCXXFLAGS .= " -Xclang -fallow-half-arguments-and-returns -D__HIP_HCC_COMPAT_MODE__=1";
    }
} elsif ($HIP_PLATFORM eq "nvidia") {
    $CUDA_PATH=$ENV{'CUDA_PATH'} // '/usr/local/cuda';
    $HIP_INCLUDE_PATH = "$HIP_PATH/include";
    if ($verbose & 0x2) {
        print ("CUDA_PATH=$CUDA_PATH\n");
    }

    $HIPCC=get_normalized_path("$CUDA_PATH/bin/nvcc");
    $HIPCXXFLAGS .= " -Wno-deprecated-gpu-targets ";
    $HIPCXXFLAGS .= " -isystem " . get_normalized_path("$CUDA_PATH/include");
    $HIPCFLAGS .= " -isystem " . get_normalized_path("$CUDA_PATH/include");

    $HIPLDFLAGS = " -Wno-deprecated-gpu-targets -lcuda -lcudart -L" . get_normalized_path("$CUDA_PATH/lib64");
} else {
    printf ("error: unknown HIP_PLATFORM = '$HIP_PLATFORM'");
    printf ("       or HIP_COMPILER = '$HIP_COMPILER'");
    exit (-1);
}

my $compileOnly = 0;
my $needCXXFLAGS = 0;  # need to add CXX flags to compile step
my $needCFLAGS = 0;    # need to add C flags to compile step
my $needLDFLAGS = 1;   # need to add LDFLAGS to compile step.
my $fileTypeFlag = 0;  # to see if -x flag is mentioned
my $hasOMPTargets = 0; # If OMP targets is mentioned
my $hasC = 0;          # options contain a c-style file
my $hasCXX = 0;        # options contain a cpp-style file (NVCC must force recognition as GPU file)
my $hasHIP = 0;        # options contain a hip-style file (HIP-Clang must pass offloading options)
my $printHipVersion = 0;    # print HIP version
my $printCXXFlags = 0;      # print HIPCXXFLAGS
my $printLDFlags = 0;       # print HIPLDFLAGS
my $runCmd = 1;
my $buildDeps = 0;
my $hsacoVersion = 0;
my $funcSupp = 0;      # enable function support
my $rdc = 0;           # whether -fgpu-rdc is on

my @options = ();
my @inputs  = ();

if ($verbose & 0x4) {
    print "hipcc-args: ", join (" ", @ARGV), "\n";
}

# Handle code object generation
my $ISACMD="";
if($HIP_PLATFORM eq "nvidia"){
    $ISACMD .= "$HIP_PATH/bin/hipcc -ptx ";
    if($ARGV[0] eq "--genco"){
        foreach $isaarg (@ARGV[1..$#ARGV]){
            $ISACMD .= " ";
            # ignore --rocm-path=xxxx on nvcc nvidia platform
            if ($isaarg !~ /--rocm-path/) {
              $ISACMD .= $isaarg;
            } else {
              print "Ignoring --rocm-path= on nvidia nvcc platform.\n";
            }
        }
        if ($verbose & 0x1) {
            print "hipcc-cmd: ", $ISACMD, "\n";
        }
        system($ISACMD) and die();
        exit(0);
    }
}

# TODO: convert toolArgs to an array rather than a string
my $toolArgs = "";  # arguments to pass to the clang or nvcc tool
my $optArg = ""; # -O args

# TODO: hipcc uses --amdgpu-target for historical reasons. It should be replaced
# by clang option --offload-arch.
my @targetOpts = ('--offload-arch=', '--amdgpu-target=');

my $targetsStr = "";
my $skipOutputFile = 0; # file followed by -o should not contibute in picking compiler flags
my $prevArg = ""; # previous argument

foreach $arg (@ARGV)
{
    # Save $arg, it can get changed in the loop.
    $trimarg = $arg;
    # TODO: figure out why this space removal is wanted.
    # TODO: If someone has gone to the effort of quoting the spaces to the shell
    # TODO: why are we removing it here?
    $trimarg =~ s/^\s+|\s+$//g;  # Remive whitespace
    my $swallowArg = 0;
    my $escapeArg = 1;
    if ($arg eq '-c' or $arg eq '--genco' or $arg eq '-E') {
        $compileOnly = 1;
        $needLDFLAGS  = 0;
    }

    if ($skipOutputFile) {
	# TODO: handle filename with shell metacharacters
        $toolArgs .= " " . get_normalized_path("$arg");
        $prevArg = $arg;
        $skipOutputFile = 0;
        next;
    }

    if ($arg eq '-o') {
        $needLDFLAGS = 1;
        $skipOutputFile = 1;
    }

    if(($trimarg eq '-stdlib=libc++') and ($setStdLib eq 0))
    {
        $HIPCXXFLAGS .= " -stdlib=libc++";
        $setStdLib = 1;
    }

    # Check target selection option: --offload-arch= and --amdgpu-target=...
    foreach my $targetOpt (@targetOpts) {
        if (substr($arg, 0, length($targetOpt)) eq $targetOpt) {
             if ($targetOpt eq '--amdgpu-target=') {
                print "Warning: The --amdgpu-target option has been deprecated and will be removed in the future.  Use --offload-arch instead.\n";
             }
             # If targets string is not empty, add a comma before adding new target option value.
             $targetsStr .= ($targetsStr ? ',' : '');
             $targetsStr .=  substr($arg, length($targetOpt));
             $default_amdgpu_target = 0;
             # Collect the GPU arch options and pass them to clang later.
             if ($HIP_PLATFORM eq "amd") {
                 $swallowArg = 1;
             }
        }
    }

    if (($arg =~ /--genco/) and  $HIP_PLATFORM eq 'amd' ) {
        $arg = "--cuda-device-only";
    }

    if($trimarg eq '--version') {
        $printHipVersion = 1;
    }
    if($trimarg eq '--short-version') {
        $printHipVersion = 1;
        $runCmd = 0;
    }
    if($trimarg eq '--cxxflags') {
        $printCXXFlags = 1;
        $runCmd = 0;
    }
    if($trimarg eq '--ldflags') {
        $printLDFlags = 1;
        $runCmd = 0;
    }
    if($trimarg eq '-M') {
        $compileOnly = 1;
        $buildDeps = 1;
    }
    if($trimarg eq '-use-staticlib') {
        print "Warning: The -use-staticlib option has been deprecated and is no longer needed.\n"
    }
    if($trimarg eq '-use-sharedlib') {
        print "Warning: The -use-sharedlib option has been deprecated and is no longer needed.\n"
    }
    if($arg =~ m/^-O/)
    {
        $optArg = $arg;
    }
    if($arg =~ '--amdhsa-code-object-version=')
    {
        print "Warning: The --amdhsa-code-object-version option has been deprecated and will be removed in the future.  Use -mllvm -mcode-object-version instead.\n";
        $arg =~ s/--amdhsa-code-object-version=//;
        $hsacoVersion = $arg;
        $swallowArg = 1;
    }

    # nvcc does not handle standard compiler options properly
    # This can prevent hipcc being used as standard CXX/C Compiler
    # To fix this we need to pass -Xcompiler for options
    if (($arg eq '-fPIC' or $arg =~ '-Wl,') and $HIP_COMPILER eq 'nvcc')
    {
        $HIPCXXFLAGS .= " -Xcompiler ".$arg;
        $swallowArg = 1;
    }

    if ($arg eq '-x') {
        $fileTypeFlag = 1;
    } elsif (($arg eq 'c' and $prevArg eq '-x') or ($arg eq '-xc')) {
        $fileTypeFlag = 1;
        $hasC = 1;
        $hasCXX = 0;
        $hasHIP = 0;
    } elsif (($arg eq 'c++' and $prevArg eq '-x') or ($arg eq '-xc++')) {
        $fileTypeFlag = 1;
        $hasC = 0;
        $hasCXX = 1;
        $hasHIP = 0;
    } elsif (($arg eq 'hip' and $prevArg eq '-x') or ($arg eq '-xhip')) {
        $fileTypeFlag = 1;
        $hasC = 0;
        $hasCXX = 0;
        $hasHIP = 1;
    } elsif ($arg  =~ '-fopenmp-targets=') {
        $hasOMPTargets = 1;
    } elsif ($arg =~ m/^-/) {
        # options start with -
        if  ($arg eq '-fgpu-rdc') {
            $rdc = 1;
        } elsif ($arg eq '-fno-gpu-rdc') {
            $rdc = 0;
        }

        # Process HIPCC options here:
        if ($arg =~ m/^--hipcc/) {
            $swallowArg = 1;
            if ($arg eq "--hipcc-func-supp") {
              print "Warning: The --hipcc-func-supp option has been deprecated and will be removed in the future.\n";
              $funcSupp = 1;
            } elsif ($arg eq "--hipcc-no-func-supp") {
              print "Warning: The --hipcc-no-func-supp option has been deprecated and will be removed in the future.\n";
              $funcSupp = 0;
            }
        } else {
            push (@options, $arg);
        }
        #print "O: <$arg>\n";
    } elsif ($prevArg ne '-o') {
        # input files and libraries
        # Skip guessing if `-x {c|c++|hip}` is already specified.

        # Add proper file extension before each file type
        # File Extension                 -> Flag
        # .c                             -> -x c
        # .cpp/.cxx/.cc/.cu/.cuh/.hip    -> -x hip
        if ($fileTypeFlag eq 0) {
            if ($arg =~ /\.c$/) {
                $hasC = 1;
                $needCFLAGS = 1;
                $toolArgs .= " -x c";
            } elsif (($arg =~ /\.cpp$/) or ($arg =~ /\.cxx$/) or ($arg =~ /\.cc$/) or ($arg =~ /\.C$/)) {
                $needCXXFLAGS = 1;
                if ($HIP_COMPILE_CXX_AS_HIP eq '0' or $HIP_PLATFORM ne "amd" or $hasOMPTargets eq 1) {
                    $hasCXX = 1;
                    if ($HIP_PLATFORM eq "nvidia") {
                        $toolArgs .= " -x cu";
                    }
                } elsif ($HIP_PLATFORM eq "amd") {
                    $hasHIP = 1;
                    $toolArgs .= " -x hip";
                }
            } elsif ((($arg =~ /\.cu$/ or $arg =~ /\.cuh$/) and $HIP_COMPILE_CXX_AS_HIP ne '0') or ($arg =~ /\.hip$/)) {
                $needCXXFLAGS = 1;
                if ($HIP_PLATFORM eq "amd") {
                    $hasHIP = 1;
                    $toolArgs .= " -x hip";
                } elsif ($HIP_PLATFORM eq "nvidia") {
                    $toolArgs .= " -x cu";
                }
            }
        }
        if ($hasC) {
            $needCFLAGS = 1;
        } elsif ($hasCXX or $hasHIP) {
            $needCXXFLAGS = 1;
        }
        push (@inputs, $arg);
        #print "I: <$arg>\n";
    }
    # Produce a version of $arg where characters significant to the shell are
    # quoted. One could quote everything of course but don't bother for
    # common characters such as alphanumerics.
    # Do the quoting here because sometimes the $arg is changed in the loop
    # Important to have all of '-Xlinker' in the set of unquoted characters.
    if (not $isWindows and $escapeArg) {
        $arg =~ s/[^-a-zA-Z0-9_=+,.\/]/\\$&/g;
    }
    $toolArgs .= " $arg" unless $swallowArg;
    $prevArg = $arg;
}

if($HIP_PLATFORM eq "amd"){
    # No AMDGPU target specified at commandline. So look for HCC_AMDGPU_TARGET
    if($default_amdgpu_target eq 1) {
        if (defined $ENV{HCC_AMDGPU_TARGET}) {
            $targetsStr = $ENV{HCC_AMDGPU_TARGET};
        } elsif (not $isWindows) {
            # Else try using rocm_agent_enumerator
            $ROCM_AGENT_ENUM = "${ROCM_PATH}/bin/rocm_agent_enumerator";
            $targetsStr = `${ROCM_AGENT_ENUM} -t GPU`;
            $targetsStr =~ s/\n/,/g;
        }
        $default_amdgpu_target = 0;
    }

    # Parse the targets collected in targetStr and set corresponding compiler options.
    my @targets = split(',', $targetsStr);
    $GPU_ARCH_OPT = " --offload-arch=";

    foreach my $val (@targets) {
        # Ignore 'gfx000' target reported by rocm_agent_enumerator.
        if ($val ne 'gfx000') {
            my @procAndFeatures = split(':', $val);
            $len = scalar @procAndFeatures;
            my $procName;
            if($len ge 1 and $len le 3) { # proc and features
                $procName = $procAndFeatures[0];
                for my $i (1 .. $#procAndFeatures) {
                    if (grep($procAndFeatures[$i], @knownFeatures) eq 0) {
                        print "Warning: The Feature: $procAndFeatures[$i] is unknown. Correct compilation is not guaranteed.\n";
                    }
                }
            } else {
                $procName = $val;
            }
            $GPU_ARCH_ARG = $GPU_ARCH_OPT . $val;
            $HIPLDARCHFLAGS .= $GPU_ARCH_ARG;
            if ($HIP_PLATFORM eq 'amd' and $hasHIP) {
                $HIPCXXFLAGS .= $GPU_ARCH_ARG;
            }
        }
    }
    if ($hsacoVersion > 0) {
        if ($compileOnly eq 0) {
            $HIPLDFLAGS .= " -mcode-object-version=$hsacoVersion";
        } else {
            $HIPCXXFLAGS .= " -mcode-object-version=$hsacoVersion";
        }
    }

    # rocm_agent_enumerator failed! Throw an error and die if linking is required
    if ($default_amdgpu_target eq 1 and $compileOnly eq 0) {
        print "No valid AMD GPU target was either specified or found. Please specify a valid target using --offload-arch=<target>.\n" and die();
    }

    $ENV{HCC_EXTRA_LIBRARIES}="\n";
}

if ($buildDeps and $HIP_PLATFORM eq 'nvidia') {
    $HIPCXXFLAGS .= " -M -D__CUDACC__";
    $HIPCFLAGS .= " -M -D__CUDACC__";
}

if ($buildDeps and $HIP_PLATFORM eq 'amd') {
    $HIPCXXFLAGS .= " --cuda-host-only";
}

# hipcc currrently requires separate compilation of source files, ie it is not possible to pass
# CPP files combined with .O files
# Reason is that NVCC uses the file extension to determine whether to compile in CUDA mode or
# pass-through CPP mode.

if ($HIP_PLATFORM eq "amd") {
    # Set default optimization level to -O3 for hip-clang.
    if ($optArg eq "") {
        $HIPCXXFLAGS .= " -O3";
        $HIPCFLAGS .= " -O3";
        $HIPLDFLAGS .= " -O3";
    }
    if (!$funcSupp and $optArg ne "-O0" and $hasHIP) {
        $HIPCXXFLAGS .= " -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false";
        if ($needLDFLAGS and not $needCXXFLAGS) {
            $HIPLDFLAGS .= " -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false";
        }
    }

    # If the HIP_PATH env var is defined, pass that path to Clang
    if ($ENV{'HIP_PATH'}) {
        my $hip_path_flag = " --hip-path=" . get_normalized_path("$HIP_PATH");
        $HIPCXXFLAGS .= $hip_path_flag;
        $HIPLDFLAGS .= $hip_path_flag;
    }

    if ($hasHIP) {
        if (defined $DEVICE_LIB_PATH) {
            $HIPCXXFLAGS .= " --hip-device-lib-path=" . get_normalized_path("$DEVICE_LIB_PATH");
        }
    }

    if (!$compileOnly) {
        $HIPLDFLAGS .= " --hip-link";
        if ($rdc) {
            $HIPLDFLAGS .= $HIPLDARCHFLAGS;
        }
        if (not $isWindows) {
            $HIPLDFLAGS .= " --rtlib=compiler-rt -unwindlib=libgcc";

        }
    }
}

if ($HIPCC_COMPILE_FLAGS_APPEND) {
    $HIPCXXFLAGS .= " $HIPCC_COMPILE_FLAGS_APPEND";
    $HIPCFLAGS .= " $HIPCC_COMPILE_FLAGS_APPEND";
}
if ($HIPCC_LINK_FLAGS_APPEND) {
    $HIPLDFLAGS .= " $HIPCC_LINK_FLAGS_APPEND";
}

# TODO: convert CMD to an array rather than a string
my $CMD="$HIPCC";

if ($needCFLAGS) {
    $CMD .= " $HIPCFLAGS";
}

if ($needCXXFLAGS) {
    $CMD .= " $HIPCXXFLAGS";
}

if ($needLDFLAGS and not $compileOnly) {
    $CMD .= " $HIPLDFLAGS";
}
$CMD .= " $toolArgs";

if ($verbose & 0x1) {
    print "hipcc-cmd: ", $CMD, "\n";
}

if ($printHipVersion) {
    if ($runCmd) {
        print "HIP version: "
    }
    print $HIP_VERSION, "\n";
}
if ($printCXXFlags) {
    print $HIPCXXFLAGS;
}
if ($printLDFlags) {
    print $HIPLDFLAGS;
}
if ($runCmd) {
    system ("$CMD");
    if ($? == -1) {
        print "failed to execute: $!\n";
        exit($?);
    }
    elsif ($? & 127) {
        printf "child died with signal %d, %s coredump\n",
        ($? & 127),  ($? & 128) ? 'with' : 'without';
        exit($?);
    }
    else {
         $CMD_EXIT_CODE = $? >> 8;
    }
    $? or delete_temp_dirs ();
    exit($CMD_EXIT_CODE);
}

# vim: ts=4:sw=4:expandtab:smartindent
