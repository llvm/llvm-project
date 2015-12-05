#!/usr/bin/perl

# This script will take a number ($ENV{SCRIPT_INPUT_FILE_COUNT}) of static archive files
# and pull them apart into object files. These object files will be placed in a directory
# named the same as the archive itself without the extension. Each object file will then
# get renamed to start with the archive name and a '-' character (for archive.a(object.o)
# the object file would becomde archive-object.o. Then all object files are re-made into
# a single static library. This can help avoid name collisions when different archive
# files might contain object files with the same name.

use strict;
use Config;
use Cwd 'abs_path';
use File::Basename;
use File::Glob ':glob';
use File::Slurp;
use List::Util qw[min max];
use Digest::MD5 qw(md5_hex);

our $llvm_srcroot = $ENV{LLVM_SOURCE_DIR};
our $llvm_dstroot = $ENV{LLVM_BUILD_DIR};
our $llvm_configuration = $ENV{LLVM_CONFIGURATION};
our $archive_filelist_file = "$ENV{LLVM_BUILD_DIR}/archives.txt";

our $llvm_configuration = $ENV{LLVM_CONFIGURATION};
our $lldb_configuration = $ENV{CONFIGURATION};

# these values only matter in submission branches
# they are provided by whoever hands out swiftlang tags
our $swift_version = "600.04.10";
our $clang_version = "600.19.85";

our $llvm_revision = "swift-master";
our $clang_revision = "swift-master";
our $compiler_rt_revision = "master";
our $swift_revision = "master";

our $lldb_version = $ENV{CURRENT_PROJECT_VERSION};

our $SRCROOT = "$ENV{SRCROOT}";
our @archs = split (/\s+/, $ENV{ARCHS});
my $os_release = 11;

our $is_ios_build = 0;
foreach my $tmparch (@archs)
{
    if ($tmparch =~ /^arm/)
    {
        $is_ios_build = 1;
    }
}

my $original_env_path = $ENV{PATH};

# xcrun --sdk macosx.internal --show-sdk-path
our $SDKROOT = "$ENV{SDKROOT}";

our $CMAKE_C_COMPILER = "$ENV{CMAKE_C_COMPILER}";

if ($CMAKE_C_COMPILER eq "")
{
    $CMAKE_C_COMPILER = `xcrun --toolchain default -f cc`;
    chomp($CMAKE_C_COMPILER);
}

our $CMAKE_CXX_COMPILER = "$ENV{CMAKE_CXX_COMPILER}";

if ($CMAKE_CXX_COMPILER eq "")
{
    $CMAKE_CXX_COMPILER = `xcrun --toolchain default -f c++`;
    chomp($CMAKE_CXX_COMPILER);
}



my $common_configure_options = "";
my $common_impl_options = "";

our %llvm_config_info;

#if (($ENV{DT_VARIANT} eq "PONDEROSA") || ($ENV{RC_APPLETV} eq "YES") || ($ENV{PLATFORM_NAME} =~ /tvos/i) || ($ENV{RC_PLATFORM_NAME} =~ /tvos/i)) {
    %llvm_config_info = (
        'Debug'           => { configure_options => '--preset=LLDB_Swift_Debug swift_install_destdir=${llvm_dstroot}/swift-macosx-x86_64'},
        'DebugAssert'   => { configure_options => '--preset=LLDB_Swift_DebugAssert swift_install_destdir=${llvm_dstroot}/swift-macosx-x86_64' },
        'DebugPresubmission'   => { configure_options => '--preset=LLDB_Swift_DebugPresubmission swift_install_destdir=${llvm_dstroot}/swift-macosx-x86_64' },
        'Release'         => { configure_options => '--preset=LLDB_Swift_Release swift_install_destdir=${llvm_dstroot}/swift-macosx-x86_64'},
        'ReleaseDebug'   => { configure_options => '--preset=LLDB_Swift_ReleaseDebug swift_install_destdir=${llvm_dstroot}/swift-macosx-x86_64'},
        'RelWithDebInfo' => { configure_options => '--preset=LLDB_Swift_ReleaseDebug swift_install_destdir=${llvm_dstroot}/swift-macosx-x86_64'},
        'RelWithDebInfoAssert' => { configure_options => '--preset=LLDB_Swift_ReleaseAssert swift_install_destdir=${llvm_dstroot}/swift-macosx-x86_64' },
        'ReleaseAssert' => { configure_options => '--preset=LLDB_Swift_ReleaseAssert swift_install_destdir=${llvm_dstroot}/swift-macosx-x86_64' },
    
        ### Don't change the --preset=LLDB_BNI bit below without also updating the is_ios_build 
        ### block below that modifies it for iOS builds.
        'BuildAndIntegration' => { configure_options => "--preset=LLDB_BNI swift_compiler_version=${swift_version} clang_tag=${clang_version} swift_tag=${swift_version} swift_install_destdir=${llvm_dstroot}/swift-macosx-x86_64" },
    );
#} else {
#    %llvm_config_info = (
#        'Debug'           => { configure_options => '--preset=LLDB_Swift_Debug_bleached'},
#        'DebugAssert'   => { configure_options => '--preset=LLDB_Swift_DebugAssert_bleached' },
#        'DebugPresubmission'   => { configure_options => '--preset=LLDB_Swift_DebugPresubmission' },
#        'Release'         => { configure_options => '--preset=LLDB_Swift_Release_bleached'},
#        'ReleaseDebug'   => { configure_options => '--preset=LLDB_Swift_ReleaseDebug_bleached'},
#        'ReleaseAssert' => { configure_options => '--preset=LLDB_Swift_ReleaseAssert_bleached' },
#    
#        ### Don't change the --preset=LLDB_BNI bit below without also updating the is_ios_build 
#        ### block below that modifies it for iOS builds.
#        'BuildAndIntegration' => { configure_options => "--preset=LLDB_BNI_bleached swift_compiler_version=${swift_version} clang_tag=${clang_version} swift_tag=${swift_version} swift_install_destdir=${llvm_dstroot}/swift-macosx-x86_64" },
#    );
#}

our %llvm_build_dirs = (
    'Debug' => 'Debug',
    'DebugAssert' => 'DebugAssert',
    'DebugPresubmission' => 'DebugAssert',
    'Release' => 'RelWithDebInfo',
    'ReleaseDebug' => 'RelWithDebInfo',
    'ReleaseAssert' => 'RelWithDebInfoAssert',
    'BuildAndIntegration' => 'RelWithDebInfo'
);

our $llvm_build_dir = "";
our $is_swift_prebuilt = 0;

if(defined $ENV{LLDB_PATH_TO_LLVM_BUILD})
{
    $llvm_build_dir = $ENV{LLDB_PATH_TO_LLVM_BUILD};
    print "swiftlang root in $llvm_build_dir\n";
    $is_swift_prebuilt = 1;
}
else
{
    $llvm_build_dir = $llvm_build_dirs{"$llvm_configuration"};
}

my $llvm_hash_includes_diffs = 0;

our $llvm_config_href = undef;
if (exists $llvm_config_info{"$llvm_configuration"})
{
    $llvm_config_href = $llvm_config_info{$llvm_configuration};
    if ($is_ios_build)
    {
        my $target_configs = "";
        my $arm_archs = join ("_", sort (@archs));
        if ($ENV{RC_PLATFORM_NAME} =~ /tvos/i || $ENV{PLATFORM_NAME} =~ /tvos/i)
        {
            $llvm_config_href->{configure_options} =~ s/LLDB_BNI /LLDB_BNI_tvos /;
            $llvm_config_href->{configure_options} =~ s/LLDB_Swift_Release$/LLDB_Swift_Release_tvos /;
            $llvm_config_href->{configure_options} =~ s/LLDB_Swift_ReleaseAssert$/LLDB_Swift_ReleaseAssert_tvos /;
            foreach my $tmparch (@archs)
            {
                $target_configs = $target_configs . "appletvos-$tmparch ";
            }
        }
        elsif ($ENV{RC_PLATFORM_NAME} =~ /watchos/i || $ENV{PLATFORM_NAME} =~ /watchos/i)
        {
            $llvm_config_href->{configure_options} =~ s/_bleached//;
            $llvm_config_href->{configure_options} =~ s/LLDB_BNI /LLDB_BNI_watchos /;
            $llvm_config_href->{configure_options} =~ s/LLDB_Swift_Release$/LLDB_Swift_Release_watchos /;
            $llvm_config_href->{configure_options} =~ s/LLDB_Swift_ReleaseAssert$/LLDB_Swift_ReleaseAssert_watchos /;
            foreach my $tmparch (@archs)
            {
                $target_configs = $target_configs . "watchos-$tmparch ";
            }
        }
        else
        {
            $llvm_config_href->{configure_options} =~ s/_bleached//;
            $llvm_config_href->{configure_options} =~ s/LLDB_BNI /LLDB_BNI_ios /;
            $llvm_config_href->{configure_options} =~ s/LLDB_Swift_Release$/LLDB_Swift_Release_ios /;
            $llvm_config_href->{configure_options} =~ s/LLDB_Swift_ReleaseAssert$/LLDB_Swift_ReleaseAssert_ios /;
            foreach my $tmparch (@archs)
            {
                $target_configs = $target_configs . "iphoneos-$tmparch ";
            }
        }
        $target_configs =~ s/ $//;
        $llvm_config_href->{configure_options} = $llvm_config_href->{configure_options} . " cross_compile_tools_deployment_targets='$target_configs'";
    }
}
else
{
    die "Unsupported LLVM configuration: '$llvm_configuration'\n";
}

if ($is_ios_build)
{
    # in llvm/tools/swift/utils/build-presets.ini the LLDB_BNI_ios preset doesn't build with
    # debug info enabled - the build root is enormous to build all the architectures with
    # debug info.  Make sure we're looking for the correct llvm build dir here.
    if ($llvm_config_href->{configure_options} =~ /_BNI_ios/ || $llvm_config_href->{configure_options} =~ /_BNI_watchos/ || $llvm_config_href->{configure_options} =~ /_BNI_tvos/)
    {
        $llvm_build_dir =~ s/RelWithDebInfo/Release/;
    }
}

our @llvm_repositories = (
    abs_path("$llvm_srcroot"),
    abs_path("$llvm_srcroot/tools/clang"),
    abs_path("$llvm_srcroot/projects/compiler-rt"),
    abs_path("$llvm_srcroot/tools/swift")
);

# Get rid of the "/BuildAndIntegration" part, use Ninja-* name
if ($is_swift_prebuilt)
{
    $llvm_dstroot = $llvm_build_dir;
}
else
{
    my $ninja_dirname = dirname ($llvm_dstroot);
    $llvm_dstroot = $ninja_dirname . "/Ninja-${llvm_build_dir}";
}

if ((-e "$llvm_srcroot/lib" ) and (-e "$llvm_srcroot/tools/clang") and (-e "$llvm_srcroot/tools/swift"))
{
    print "Using existing llvm sources in: '$llvm_srcroot'\n";
    print "Using standard LLVM build directory:\n  SRC = '$llvm_srcroot'\n  DST = '$llvm_dstroot'\n";
}
elsif ($is_swift_prebuilt)
{
    print "Using prebuilt llvm in: ${llvm_build_dir}"
}
else
{
    print "Checking out llvm sources...\n";
    do_command ("cd '$SRCROOT' && git clone ssh://git\@github.com/apple/swift-llvm.git llvm", "checking out llvm from repository", 1);
    do_command("pushd $SRCROOT/llvm; git fetch --all --tags; git checkout $llvm_revision; popd",1);
    print "Checking out clang sources...\n";
    do_command ("cd '$SRCROOT/llvm/tools' && git clone ssh://git\@github.com/apple/swift-clang.git clang", "checking out clang from repository", 1);
    do_command("pushd $SRCROOT/llvm/tools/clang; git fetch --all --tags; git checkout $clang_revision; popd",1);

# compiler_rt will not build (currently, Feb 2015) for arm; don't check it out
# for native builds which might be re-used for arm.  It's only needed for running
# the testsuite while using the self-built clang as a compiler for the ASAN tests.
#    if (!$is_ios_build)
#    {
#        print "Checking out compiler-rt sources from revision $compiler_rt_revision...\n";
#        do_command ("cd '$SRCROOT/llvm/projects' && git clone ssh://git\@github.com/apple/swift-compiler-rt.git compiler-rt", "checking out compiler-rt from repository", 1);
#        do_command("pushd $SRCROOT/llvm/projects/compiler-rt; git fetch --all --tags; git checkout $compiler_rt_revision; popd",1);
#    }

    print "Checking out swift sources...\n";
    do_command ("cd '$SRCROOT/llvm/tools' && git clone ssh://git\@github.com/apple/swift.git", "checking out swift from repository", 1);
    do_command("pushd $SRCROOT/llvm/tools/swift; git fetch --all; git checkout $swift_revision; popd",1);

    print "Checking out ninja sources... \n";
    do_command("cd '$SRCROOT' && git clone git://github.com/martine/ninja.git", "checking out ninja from repository", 0);

    print "Applying any local patches to LLVM/Clang...";

    my @llvm_patches = bsd_glob("$ENV{SRCROOT}/scripts/llvm.*.diff");
    foreach my $patch (@llvm_patches)
    {
        do_command ("cd '$llvm_srcroot' && patch -p0 < $patch");
    }

    my @clang_patches = bsd_glob("$ENV{SRCROOT}/scripts/clang.*.diff");
    foreach my $patch (@clang_patches)
    {
        do_command ("cd '$llvm_srcroot/tools/clang' && patch -p0 < $patch");
    }

    my @compiler_rt_patches = bsd_glob("$ENV{SRCROOT}/scripts/compiler-rt.*.diff");
    foreach my $patch (@compiler_rt_patches)
    {
        do_command ("cd '$llvm_srcroot/projects/compiler-rt' && patch -p0 < $patch");
    }

    my @swift_patches = bsd_glob("$ENV{SRCROOT}/scripts/swift.*.diff");
    foreach my $patch (@swift_patches)
    {
        do_command ("cd '$llvm_srcroot/tools/swift' && patch -p0 < $patch");
    }
}

if (not $is_swift_prebuilt)
{
    # if an existing clang or swift symlink is lurking around remove it before adding it anew
    # this helps prevent B&I issues if one forgets to remove their local symlinks before submitting
    if ((-e "$SRCROOT/clang"))
    {
        do_command("rm $SRCROOT/clang", "remove existing clang symlink or whatnot", 0);
    }
    do_command("cd $SRCROOT; ln -s llvm/tools/clang clang", "symlinking clang", 0);
    
    if ((-e "$SRCROOT/swift"))
    {
        do_command("rm $SRCROOT/swift", "remove existing swift symlink or whatnot", 0);
    }
    do_command("cd $SRCROOT; ln -s llvm/tools/swift swift", "symlinking swift", 0);
}

# Get our options

our $debug = 1;

sub parallel_guess
{
    my $cpus = `sysctl -n hw.ncpu`;
    chomp ($cpus);
    my $memsize = `sysctl -n hw.memsize`;
    chomp ($memsize);
    my $max_cpus_by_memory = int($memsize / (750 * 1024 * 1024));
    return min($max_cpus_by_memory, $cpus);
}

sub build_llvm
{
    #my $extra_svn_options = $debug ? "" : "--quiet";
    # Make the llvm build directory
    my $arch_idx = 0;
    
    # Calculate if the current source digest so we can compare it to each architecture
    # build folder
    my @llvm_md5_strings;
    foreach my $repo (@llvm_repositories)
    {
        if (-d "$repo/.svn")
        {
            push(@llvm_md5_strings, `cd $repo; svn info`);
            if ($llvm_hash_includes_diffs == 1)
            {
                push(@llvm_md5_strings, `cd $repo; svn diff`);
            }
        }
        elsif (-d "$repo/.git")
        {
            push(@llvm_md5_strings, `cd '$repo'; git branch -v`);
            if ($llvm_hash_includes_diffs == 1)
            {
                push(@llvm_md5_strings, `cd '$repo'; git diff`);
            }
        }
    }
    
    # open my $md5_data_file, '>', "/tmp/a.txt" or die "Can't open $! for writing...\n";
    # foreach my $md5_string (@llvm_md5_strings)
    # {
    #     print $md5_data_file $md5_string;
    # }
    # close ($md5_data_file);
    
    #print "LLVM MD5 will be generated from:\n";
    #print @llvm_md5_strings;
    my $llvm_hex_digest = md5_hex(@llvm_md5_strings);
    my $did_make = 0;
    
    #print "llvm MD5: $llvm_hex_digest\n";

    my @archive_dirs;

    foreach my $arch (@archs)
    {

        # if the arch destination root exists we have already built it
        my $do_configure = 0;
        my $do_make = 0;
        my $save_arch_digest = 1;

        my $llvm_dstroot_arch = "${llvm_dstroot}/llvm-macosx-x86_64";
        my $swift_dstroot_arch = "${llvm_dstroot}/swift-macosx-x86_64";
        my $cmark_dstroot_arch = "${llvm_dstroot}/cmark-macosx-x86_64";

        if ($is_ios_build)
        {
            if ($ENV{RC_PLATFORM_NAME} =~ /tvos/i || $ENV{PLATFORM_NAME} =~ /tvos/i)
            {
                $llvm_dstroot_arch = "${llvm_dstroot}/llvm-appletvos-$arch";
                $swift_dstroot_arch = "${llvm_dstroot}/swift-appletvos-$arch";
                $cmark_dstroot_arch = "${llvm_dstroot}/cmark-appletvos-$arch";
            }
            elsif ($ENV{RC_PLATFORM_NAME} =~ /watchos/i || $ENV{PLATFORM_NAME} =~ /watchos/i)
            {
                $llvm_dstroot_arch = "${llvm_dstroot}/llvm-watchos-$arch";
                $swift_dstroot_arch = "${llvm_dstroot}/swift-watchos-$arch";
                $cmark_dstroot_arch = "${llvm_dstroot}/cmark-watchos-$arch";
            }
            else
            {
                $llvm_dstroot_arch = "${llvm_dstroot}/llvm-iphoneos-$arch";
                $swift_dstroot_arch = "${llvm_dstroot}/swift-iphoneos-$arch";
                $cmark_dstroot_arch = "${llvm_dstroot}/cmark-iphoneos-$arch";
            }
        }

        if ($is_swift_prebuilt)
        {
            # a prebuilt swiftlang has been already updated by swift's build logic
            $llvm_dstroot_arch = "$ENV{LLDB_PATH_TO_LLVM_BUILD}";
            $swift_dstroot_arch = "$ENV{LLDB_PATH_TO_SWIFT_BUILD}";
            $cmark_dstroot_arch = "$ENV{LLDB_PATH_TO_CMARK_BUILD}";

            push @archive_dirs, "$llvm_dstroot_arch/lib";
            push @archive_dirs, "$swift_dstroot_arch/lib";
            push @archive_dirs, "$cmark_dstroot_arch/src";

            print "Prebuilt swiftlang exists at $llvm_dstroot_arch + $swift_dstroot_arch + $cmark_dstroot_arch\n";

            next;
        }
        else
        {
            push @archive_dirs, "$llvm_dstroot_arch/lib";
            push @archive_dirs, "$swift_dstroot_arch/lib";
            push @archive_dirs, "$cmark_dstroot_arch/src";
        }        

        my $arch_digest_file = "${llvm_dstroot}/${arch}.md5";

        print "LLVM architecture root for ${arch} exists at '$llvm_dstroot_arch'...";
        if ((-e $llvm_dstroot_arch) and (-e $swift_dstroot_arch))
        {
            print " YES\n";
            
            $do_configure = !-e "$llvm_dstroot_arch/CMakeCache.txt";

            my @archive_modtimes;
            if ($do_make == 0)
            {
                if (-e $arch_digest_file)
                {
                    my $arch_hex_digest = read_file($arch_digest_file);
                    if ($arch_hex_digest eq $llvm_hex_digest)
                    {
                        # No sources have been changed or updated
                        $save_arch_digest = 0;
                    }
                    else
                    {
                        # Sources have changed, or svn has been updated
                        print "Sources have changed, rebuilding...\n";
                        $do_make = 1;
                    }
                }
                else
                {
                    # No MD5 digest, we need to make
                    print "Missing MD5 digest file '$arch_digest_file', rebuilding...\n";
                    $do_make = 1;
                }
                
                if ($do_make == 0)
                {
                    if (-e $archive_filelist_file)
                    {
                        # the final archive exists, check the modification times on all .a files that
                        # make the final archive to make sure we don't need to rebuild
                        my $archive_filelist_file_modtime = (stat($archive_filelist_file))[9];
                        
                        our @archive_files = glob "$llvm_dstroot_arch/lib/*.a";
                        push @archive_files, glob "$swift_dstroot_arch/lib/*.a";
                        push @archive_files, glob "$cmark_dstroot_arch/src/*.a";
                        
                        for my $llvm_lib (@archive_files)
                        {
                            if (-e $llvm_lib)
                            {
                                if ($archive_filelist_file_modtime < (stat($llvm_lib))[9])
                                {
                                    print "'$llvm_dstroot_arch/$llvm_lib' is newer than '$archive_filelist_file', rebuilding...\n";
                                    $do_make = 1;
                                    last;
                                }
                            }
                        }
                    }
                    else
                    {
                        $do_make = 1;
                    }
                }
            }
        }
        else
        {
            print "NO\n";

            do_command ("mkdir -p '$llvm_dstroot_arch'", "making llvm build directory '$llvm_dstroot_arch'", 0);
            do_command ("mkdir -p '$swift_dstroot_arch'", "making swift build directory '$swift_dstroot_arch'", 0);
            $do_configure = 1;
            $do_make = 1;

            if ($is_ios_build)
            {
                my $llvm_dstroot_arch_bin = "${llvm_dstroot_arch}/bin";
                if (!-d $llvm_dstroot_arch_bin)
                {
                    do_command ("mkdir -p '$llvm_dstroot_arch_bin'", "making llvm build arch bin directory '$llvm_dstroot_arch_bin'", 0);
                    my @tools = ("ar", "nm", "strip", "lipo", "ld", "as");
                    my $script_mode = 0755;
                    my $prog;
                    for $prog (@tools)
                    {
                        chomp(my $actual_prog_path = `xcrun -sdk '$SDKROOT' -find ${prog}`);
                        symlink($actual_prog_path, "$llvm_dstroot_arch_bin/${prog}");
                        my $script_prog_path = "$llvm_dstroot_arch_bin/arm-apple-darwin${os_release}-${prog}";
                        open (SCRIPT, ">$script_prog_path") or die "Can't open $! for writing...\n";
                        print SCRIPT "#!/bin/sh\nexec '$actual_prog_path' \"\$\@\"\n";
                        close (SCRIPT);
                        chmod($script_mode, $script_prog_path);
                    }
                    #  Tools that must have the "-arch" and "-sysroot" specified
                    my @arch_sysroot_tools = ("clang", "clang++", "gcc", "g++");
                    for $prog (@arch_sysroot_tools)
                    {
                        chomp(my $actual_prog_path = `xcrun -sdk '$SDKROOT' -find ${prog}`);
                        symlink($actual_prog_path, "$llvm_dstroot_arch_bin/${prog}");
                        my $script_prog_path = "$llvm_dstroot_arch_bin/arm-apple-darwin${os_release}-${prog}";
                        open (SCRIPT, ">$script_prog_path") or die "Can't open $! for writing...\n";
                        print SCRIPT "#!/bin/sh\nexec '$actual_prog_path' -arch ${arch} -isysroot '$SDKROOT' \"\$\@\"\n";
                        close (SCRIPT);
                        chmod($script_mode, $script_prog_path);
                    }
                    my $new_path = "$original_env_path:$llvm_dstroot_arch_bin";
                    print "Setting new environment PATH = '$new_path'\n";
                    $ENV{PATH} = $new_path;
                }
            }
        }
        
        if ($save_arch_digest)
        {
            write_file($arch_digest_file, \$llvm_hex_digest);
        }

        if ($do_make)
        {
            $do_configure = 1;
        }

        if ($do_configure)
        {
            # Build llvm and clang
            print "Configuring clang ($arch) in '$llvm_dstroot_arch'...\n";
            #per Dmitri, DLLVM_TARGETS_TO_BUILD=\"X86;ARM;AArch64\" is the default
            my $lldb_configuration_options = "$common_configure_options $llvm_config_href->{configure_options} $common_impl_options";

            # We're configuring llvm/clang with --enable-cxx11 and --enable-libcpp but llvm/configure doesn't
            # pick up the right C++ standard library.  If we have a MACOSX_DEPLOYMENT_TARGET of 10.7 or 10.8
            # (or are using actually building on those releases), we need to specify "-stdlib=libc++" at link
            # time or llvm/configure will not see <atomic> as available and error out (v. llvm r199313).
            # $ENV{LDFLAGS} = $ENV{LDFLAGS} . " -stdlib=libc++";

            # Unset "SDKROOT" for ARM builds
            do_command ("SWIFT_SOURCE_ROOT=$ENV{SRCROOT} SWIFT_BUILD_ROOT=$ENV{LLVM_BUILD_DIRTREE} ./swift/utils/build-script $lldb_configuration_options",
	                        "configuring llvm build (Unix Makefiles)", 1);

        }

        if ($do_make)
        {
            $did_make = 1;
            # Build llvm and clang
            # These all seem redundant
            # Combine all .o files from a bunch of static libraries from llvm
            # and clang into a single .a file.
            do_command("rm -rf $ENV{LLVM_BUILD_DIR}; ln -s $ENV{LLVM_BUILD_DIRTREE}/Ninja-$llvm_build_dir $ENV{LLVM_BUILD_DIR}", "symlinking config", 0);
        }

        ++$arch_idx;
    }

    # If we did any makes update the archive filenames file with any .a files from
    # each architectures "lib" folder...
    if ($did_make || $is_swift_prebuilt)
    {
        open my $fh, '>', $archive_filelist_file or die "Can't open $! for writing...\n";
        foreach my $archive_dir (@archive_dirs)
        {
            append_all_archive_files ($archive_dir, $fh);
        }
        close($fh);
    }
}

#----------------------------------------------------------------------
# quote the path if needed and realpath it if the -r option was
# specified
#----------------------------------------------------------------------
sub finalize_path
{
    my $path = shift;
    # Realpath all paths that don't start with "/"
    $path =~ /^[^\/]/ and $path = abs_path($path);

    # Quote the path if asked to, or if there are special shell characters
    # in the path name
    my $has_double_quotes = $path =~ /["]/;
    my $has_single_quotes = $path =~ /[']/;
    my $needs_quotes = $path =~ /[ \$\&\*'"]/;
    if ($needs_quotes)
    {
        # escape and double quotes in the path
        $has_double_quotes and $path =~ s/"/\\"/g;
        $path = "\"$path\"";
    }
    return $path;
}

sub do_command
{
    my $cmd = shift;
    my $description = @_ ? shift : "command";
    my $die_on_fail = @_ ? shift : undef;
    $debug and print "% $cmd\n";
    system ($cmd);
    if ($? == -1)
    {
        $debug and printf ("error: %s failed to execute: $!\n", $description);
        $die_on_fail and $? and exit(1);
        return $?;
    }
    elsif ($? & 127)
    {
        $debug and printf("error: %s child died with signal %d, %s coredump\n",
                          $description,
                          ($? & 127),
                          ($? & 128) ? 'with' : 'without');
        $die_on_fail and $? and exit(1);
        return $?;
    }
    else
    {
        my $exit = $? >> 8;
        if ($exit)
        {
            $debug and printf("error: %s child exited with value %d\n", $description, $exit);
            $die_on_fail and exit(1);
        }
        return $exit;
    }
}

sub append_all_archive_files
{
   my $archive_dir = shift;
   my $fh = shift;

   our @archive_files = glob "$archive_dir/*.a";    
   for my $archive_fullpath (@archive_files)
   {
       print $fh "$archive_fullpath\n";
   }
}

build_llvm();
