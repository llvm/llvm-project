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
use Getopt::Long;
use Cwd;

# Return name of HIP compiler - either 'clang' or 'nvcc'
#
use Getopt::Long;
use File::Basename;

my $base_dir;
BEGIN {
    $base_dir = dirname( Cwd::realpath(__FILE__) );
}
use lib "$base_dir/";
use hipvars;

$isWindows      =   $hipvars::isWindows;
$HIP_RUNTIME    =   $hipvars::HIP_RUNTIME;
$HIP_PLATFORM   =   $hipvars::HIP_PLATFORM;
$HIP_COMPILER   =   $hipvars::HIP_COMPILER;
$HIP_CLANG_PATH =   $hipvars::HIP_CLANG_PATH;
$CUDA_PATH      =   $hipvars::CUDA_PATH;
$HIP_PATH       =   $hipvars::HIP_PATH;
$ROCM_PATH      =   $hipvars::ROCM_PATH;
$HIP_VERSION    =   $hipvars::HIP_VERSION;

Getopt::Long::Configure ( qw{bundling no_ignore_case});
GetOptions(
     "help|h" => \$p_help
    ,"path|p" => \$p_path
    ,"rocmpath|R" => \$p_rocmpath
    ,"compiler|c" => \$p_compiler
    ,"platform|P" => \$p_platform
    ,"runtime|r" => \$p_runtime
    ,"hipclangpath|l" => \$p_hipclangpath
    ,"cpp_config|cxx_config|C" => \$p_cpp_config
    ,"full|f|info" => \$p_full,
    ,"version|v" => \$p_version,
    ,"check" => \$p_check,
    ,"newline|n" => \$p_newline
);

if ($HIP_COMPILER eq "clang") {
    $HIP_CLANG_INCLUDE = "";
    if($isWindows) {
        $HIP_CLANG_INCLUDE = `\"$HIP_CLANG_PATH/clang++\" --print-resource-dir`;
    } else {
        $HIP_CLANG_INCLUDE = `$HIP_CLANG_PATH/clang++ --print-resource-dir`;
        chomp($HIP_CLANG_INCLUDE)
    }

    $CPP_CONFIG = " -D__HIP_PLATFORM_HCC__= -D__HIP_PLATFORM_AMD__=";

    $HIP_PATH_INCLUDE = $HIP_PATH."/include";
    if($isWindows) {
        $CPP_CONFIG .= " -I\"$HIP_PATH_INCLUDE\" -I\"$HIP_CLANG_INCLUDE\"";
    } else {
        $CPP_CONFIG .= " -I$HIP_PATH_INCLUDE -I$HIP_CLANG_INCLUDE ";
    }
}
if ($HIP_PLATFORM eq "nvidia") {
    $CPP_CONFIG = " -D__HIP_PLATFORM_NVCC__= -D__HIP_PLATFORM_NVIDIA__= -I$HIP_PATH/include -I$CUDA_PATH/include";
};

if ($p_help) {
    print "usage: hipconfig [OPTIONS]\n";
    print "  --path,  -p        : print HIP_PATH (use env var if set, else determine from hipconfig path)\n";
    print "  --rocmpath,  -R    : print ROCM_PATH (use env var if set, else determine from hip path or /opt/rocm)\n";
    print "  --cpp_config, -C   : print C++ compiler options\n";
    print "  --compiler, -c     : print compiler (clang or nvcc)\n";
    print "  --platform, -P     : print platform (amd or nvidia)\n";
    print "  --runtime, -r      : print runtime (rocclr or cuda)\n";
    print "  --hipclangpath, -l : print HIP_CLANG_PATH\n";
    print "  --full, -f         : print full config\n";
    print "  --version, -v      : print hip version\n";
    print "  --check            : check configuration\n";
    print "  --newline, -n      : print newline\n";
    print "  --help, -h         : print help message\n";
    exit();
}

if ($p_path) {
    print "$HIP_PATH";
    $printed = 1;
}

if ($p_rocmpath) {
    print "$ROCM_PATH";
    $printed = 1;
}

if ($p_cpp_config) {
    print $CPP_CONFIG;
    $printed = 1;
}

if ($p_compiler) {
    print $HIP_COMPILER;
    $printed = 1;
}

if ($p_platform) {
    print $HIP_PLATFORM;
    $printed = 1;
}

if ($p_runtime) {
    print $HIP_RUNTIME;
    $printed = 1;
}

if ($p_hipclangpath) {
    if (defined $HIP_CLANG_PATH) {
       print $HIP_CLANG_PATH;
    }
    $printed = 1;
}

if ($p_version) {
    print $HIP_VERSION;
    $printed = 1;
}

if (!$printed or $p_full) {
    print "HIP version  : ", $HIP_VERSION, "\n\n";
    print "== hipconfig\n";
    print "HIP_PATH     : ", $HIP_PATH, "\n";
    print "ROCM_PATH    : ", $ROCM_PATH, "\n";
    print "HIP_COMPILER : ", $HIP_COMPILER, "\n";
    print "HIP_PLATFORM : ", $HIP_PLATFORM, "\n";
    print "HIP_RUNTIME  : ", $HIP_RUNTIME, "\n";
    print "CPP_CONFIG   : ", $CPP_CONFIG, "\n";
    if ($HIP_PLATFORM eq "amd")
    {
        print "\n" ;
        if ($HIP_COMPILER eq "clang")
        {
            print "== hip-clang\n";
            print ("HIP_CLANG_PATH   : $HIP_CLANG_PATH\n");
            if ($isWindows) {
                system("\"$HIP_CLANG_PATH/clang++\" --version");
                system("\"$HIP_CLANG_PATH/llc\" --version");
                printf("hip-clang-cxxflags : ");
                $win_output = `perl \"$HIP_PATH/bin/hipcc\" --cxxflags`;
                printf("$win_output \n");
                printf("hip-clang-ldflags  : ");
                $win_output = `perl \"$HIP_PATH/bin/hipcc\" --ldflags`;
                printf("$win_output \n");
            } else {
                system("$HIP_CLANG_PATH/clang++ --version");
                system("$HIP_CLANG_PATH/llc --version");
                print ("hip-clang-cxxflags : ");
                system("$HIP_PATH/bin/hipcc --cxxflags");
                printf("\n");
                print ("hip-clang-ldflags  : ");
                system("$HIP_PATH/bin/hipcc --ldflags");
                printf("\n");
            }
        } else {
            print ("Unexpected HIP_COMPILER: $HIP_COMPILER\n");
        }
    }
    if ($HIP_PLATFORM eq "nvidia")  {
        print "\n" ;
        print "== nvcc\n";
        print "CUDA_PATH   : ", $CUDA_PATH, "\n";
        system("$CUDA_PATH/bin/nvcc --version");

    }
    print "\n" ;

    print "=== Environment Variables\n";
    if ($isWindows) {
        print ("PATH=$ENV{PATH}\n");
        system("set | findstr //B //C:\"HIP\" //C:\"CUDA\" //C:\"LD_LIBRARY_PATH\"");
    } else {
        system("echo PATH=\$PATH");
        system("env | egrep '^HIP|^CUDA|^LD_LIBRARY_PATH'");
    }


    print "\n" ;
    if ($isWindows) {
        print "== Windows Display Drivers\n";
        print "Hostname     : "; system ("hostname");
        system ("wmic path win32_VideoController get AdapterCompatibility,InstalledDisplayDrivers,Name | findstr //B //C:\"Advanced Micro Devices\"");
    } else {
        print "== Linux Kernel\n";
        print "Hostname     : "; system ("hostname");
        system ("uname -a");
    }

    if (-e "/usr/bin/lsb_release") {
        system ("/usr/bin/lsb_release -a");
    }

    print "\n" ;
    $printed = 1;
}


if ($p_check) {
    print "\nCheck system installation:\n";

    printf ("%-70s", "check hipconfig in PATH...");
    # Safer to use which hipconfig instead of invoking hipconfig
    if (system ("which hipconfig > /dev/null 2>&1") != 0)  {
        print "FAIL\n";
    } else {
        printf "good\n";
    }
}

if ($p_newline) {
    print "\n";
}
