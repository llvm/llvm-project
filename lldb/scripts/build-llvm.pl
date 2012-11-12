#!/usr/bin/perl

# This script will take a number ($ENV{SCRIPT_INPUT_FILE_COUNT}) of static archive files
# and pull them apart into object files. These object files will be placed in a directory
# named the same as the archive itself without the extension. Each object file will then
# get renamed to start with the archive name and a '-' character (for archive.a(object.o)
# the object file would becomde archive-object.o. Then all object files are re-made into
# a single static library. This can help avoid name collisions when different archive
# files might contain object files with the same name.

use strict;
use File::Basename;
use File::Glob ':glob';
use List::Util qw[min max];

our $llvm_srcroot = $ENV{SCRIPT_INPUT_FILE_0};
our $llvm_dstroot = $ENV{SCRIPT_INPUT_FILE_1};

our $llvm_clang_outfile = $ENV{SCRIPT_OUTPUT_FILE_0};
our ($llvm_clang_basename, $llvm_clang_dirname) = fileparse ($llvm_clang_outfile);

our $llvm_configuration = $ENV{LLVM_CONFIGURATION};

our $llvm_revision = "HEAD";
our $clang_revision = "HEAD";

our $SRCROOT = "$ENV{SRCROOT}";
our @archs = split (/\s+/, $ENV{ARCHS});
my $os_release = 11;

my $original_env_path = $ENV{PATH};

our %llvm_config_info = (
 'Debug'         => { configure_options => '--disable-optimized --disable-assertions --enable-libcpp', make_options => 'DEBUG_SYMBOLS=1'},
 'Debug+Asserts' => { configure_options => '--disable-optimized --enable-assertions --enable-libcpp' , make_options => 'DEBUG_SYMBOLS=1'},
 'Release'       => { configure_options => '--enable-optimized --disable-assertions --enable-libcpp' , make_options => ''},
 'Release+Debug' => { configure_options => '--enable-optimized --disable-assertions --enable-libcpp' , make_options => 'DEBUG_SYMBOLS=1'},
 'Release+Asserts' => { configure_options => '--enable-optimized --enable-assertions --enable-libcpp' , make_options => ''},
);

our $llvm_config_href = undef;
if (exists $llvm_config_info{"$llvm_configuration"})
{
	$llvm_config_href = $llvm_config_info{$llvm_configuration};
}
else
{
	die "Unsupported LLVM configuration: '$llvm_configuration'\n";
}

our @archive_files = (  
    "$llvm_configuration/lib/libclang.a",
	"$llvm_configuration/lib/libclangAnalysis.a",
	"$llvm_configuration/lib/libclangAST.a",
	"$llvm_configuration/lib/libclangBasic.a",
	"$llvm_configuration/lib/libclangCodeGen.a",
    "$llvm_configuration/lib/libclangEdit.a",
	"$llvm_configuration/lib/libclangFrontend.a",
	"$llvm_configuration/lib/libclangDriver.a",
	"$llvm_configuration/lib/libclangLex.a",
	"$llvm_configuration/lib/libclangParse.a",
	"$llvm_configuration/lib/libclangSema.a",
    "$llvm_configuration/lib/libclangSerialization.a",
	"$llvm_configuration/lib/libLLVMAnalysis.a",
	"$llvm_configuration/lib/libLLVMArchive.a",
	"$llvm_configuration/lib/libLLVMARMAsmParser.a",
	"$llvm_configuration/lib/libLLVMARMAsmPrinter.a",
	"$llvm_configuration/lib/libLLVMARMCodeGen.a",
    "$llvm_configuration/lib/libLLVMARMDesc.a",
	"$llvm_configuration/lib/libLLVMARMDisassembler.a",
	"$llvm_configuration/lib/libLLVMARMInfo.a",
	"$llvm_configuration/lib/libLLVMAsmParser.a",
	"$llvm_configuration/lib/libLLVMAsmPrinter.a",
	"$llvm_configuration/lib/libLLVMBitReader.a",
	"$llvm_configuration/lib/libLLVMBitWriter.a",
	"$llvm_configuration/lib/libLLVMCodeGen.a",
	"$llvm_configuration/lib/libLLVMCore.a",
	"$llvm_configuration/lib/libLLVMExecutionEngine.a",
	"$llvm_configuration/lib/libLLVMInstCombine.a",
	"$llvm_configuration/lib/libLLVMInstrumentation.a",
	"$llvm_configuration/lib/libLLVMipa.a",
	"$llvm_configuration/lib/libLLVMInterpreter.a",
	"$llvm_configuration/lib/libLLVMipo.a",
	"$llvm_configuration/lib/libLLVMJIT.a",
	"$llvm_configuration/lib/libLLVMLinker.a",
	"$llvm_configuration/lib/libLLVMMC.a",
	"$llvm_configuration/lib/libLLVMMCParser.a",
	"$llvm_configuration/lib/libLLVMMCDisassembler.a",
    "$llvm_configuration/lib/libLLVMMCJIT.a",
    "$llvm_configuration/lib/libLLVMObject.a",
    "$llvm_configuration/lib/libLLVMRuntimeDyld.a",
	"$llvm_configuration/lib/libLLVMScalarOpts.a",
	"$llvm_configuration/lib/libLLVMSelectionDAG.a",
	"$llvm_configuration/lib/libLLVMSupport.a",
	"$llvm_configuration/lib/libLLVMTarget.a",
	"$llvm_configuration/lib/libLLVMTransformUtils.a",
	"$llvm_configuration/lib/libLLVMX86AsmParser.a",
	"$llvm_configuration/lib/libLLVMX86AsmPrinter.a",
	"$llvm_configuration/lib/libLLVMX86CodeGen.a",
    "$llvm_configuration/lib/libLLVMX86Desc.a",
	"$llvm_configuration/lib/libLLVMX86Disassembler.a",
	"$llvm_configuration/lib/libLLVMX86Info.a",
    "$llvm_configuration/lib/libLLVMX86Utils.a",
);

if (-e "$llvm_srcroot/lib")
{
    print "Using existing llvm sources in: '$llvm_srcroot'\n";
    print "Using standard LLVM build directory:\n  SRC = '$llvm_srcroot'\n  DST = '$llvm_dstroot'\n";
}
else
{
    print "Checking out llvm sources from revision $llvm_revision...\n";
    do_command ("cd '$SRCROOT' && svn co --quiet --revision $llvm_revision http://llvm.org/svn/llvm-project/llvm/trunk llvm", "checking out llvm from repository", 1); 
    print "Checking out clang sources from revision $clang_revision...\n";
    do_command ("cd '$llvm_srcroot/tools' && svn co --quiet --revision $clang_revision http://llvm.org/svn/llvm-project/cfe/trunk clang", "checking out clang from repository", 1);
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
}

# If our output file already exists then we need not generate it again.
if (-e $llvm_clang_outfile)
{
	exit 0;
}


# Get our options

our $debug = 1;

sub parallel_guess
{
	my $cpus = `sysctl -n hw.availcpu`;
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
    foreach my $arch (@archs)
    {
        my $llvm_dstroot_arch = "${llvm_dstroot}/${arch}";

		# if the arch destination root exists we have already built it
		my $do_configure = 0;
		my $do_make = 0;
		my $is_arm = $arch =~ /^arm/;
        
		my $llvm_dstroot_arch_archive = "$llvm_dstroot_arch/$llvm_clang_basename";
		print "LLVM architecture root for ${arch} exists at '$llvm_dstroot_arch'...";
		if (-e $llvm_dstroot_arch)
		{
			print "YES\n";
			$do_configure = !-e "$llvm_dstroot_arch/config.log";
			
			# dstroot for llvm build exists, make sure all .a files are built
			for my $llvm_lib (@archive_files)
			{
				if (!-e "$llvm_dstroot_arch/$llvm_lib")
				{
					print "missing archive: '$llvm_dstroot_arch/$llvm_lib'\n";
					$do_make = 1;
				}
			}	
			if (!-e $llvm_dstroot_arch_archive)
			{
				$do_make = 1;
			}
			else
			{
				print "LLVM architecture archive for ${arch} is '$llvm_dstroot_arch_archive'\n";
			}		
		}
		else
		{
			print "NO\n";
	        do_command ("mkdir -p '$llvm_dstroot_arch'", "making llvm build directory '$llvm_dstroot_arch'", 1);
			$do_configure = 1;
			$do_make = 1;

			if ($is_arm)
			{
		        my $llvm_dstroot_arch_bin = "${llvm_dstroot_arch}/bin";

				if (!-d $llvm_dstroot_arch_bin)
				{
					do_command ("mkdir -p '$llvm_dstroot_arch_bin'", "making llvm build arch bin directory '$llvm_dstroot_arch_bin'", 1);
					my @tools = ("ar", "nm", "ranlib", "strip", "lipo", "ld", "as");
					my $script_mode = 0755;
					my $prog;
					for $prog (@tools)
					{
					   	chomp(my $actual_prog_path = `xcrun -sdk '$ENV{SDKROOT}' -find ${prog}`);
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
					   	chomp(my $actual_prog_path = `xcrun -sdk '$ENV{SDKROOT}' -find ${prog}`);
					   	my $script_prog_path = "$llvm_dstroot_arch_bin/arm-apple-darwin${os_release}-${prog}";
						open (SCRIPT, ">$script_prog_path") or die "Can't open $! for writing...\n";
						print SCRIPT "#!/bin/sh\nexec '$actual_prog_path' -arch ${arch} -isysroot '$ENV{SDKROOT}' \"\$\@\"\n";
						close (SCRIPT);
					   	chmod($script_mode, $script_prog_path);
					}
					my $new_path = "$original_env_path:$llvm_dstroot_arch_bin";
					print "Setting new environment PATH = '$new_path'\n";
			        $ENV{PATH} = $new_path;
				}
			}
		}
		
		if ($do_configure)
		{
			# Build llvm and clang
	        print "Configuring clang ($arch) in '$llvm_dstroot_arch'...\n";
			my $lldb_configuration_options = "--enable-targets=x86_64,arm $llvm_config_href->{configure_options}";

            if ($is_arm)
            {
                $lldb_configuration_options .= " --host=arm-apple-darwin${os_release} --target=arm-apple-darwin${os_release} --build=i686-apple-darwin${os_release}";
            }
            else
            {
                $lldb_configuration_options .= " --build=$arch-apple-darwin${os_release}";
            }
	        do_command ("cd '$llvm_dstroot_arch' && '$llvm_srcroot/configure' $lldb_configuration_options",
	                    "configuring llvm build", 1);			
		}

		if ($do_make)
		{
			# Build llvm and clang
			my $num_cpus = parallel_guess();
			print "Building clang using $num_cpus cpus ($arch)...\n";
            my $extra_make_flags = '';
            if ($is_arm)
            {
                $extra_make_flags = "UNIVERSAL=1 UNIVERSAL_ARCH=${arch} UNIVERSAL_SDK_PATH='$ENV{SDKROOT}'";
            }
            do_command ("cd '$llvm_dstroot_arch' && make -j$num_cpus clang-only VERBOSE=1 $llvm_config_href->{make_options} NO_RUNTIME_LIBS=1 PROJECT_NAME='llvm' $extra_make_flags", "making llvm and clang", 1);			
            do_command ("cd '$llvm_dstroot_arch' && make -j$num_cpus tools-only VERBOSE=1 $llvm_config_href->{make_options} NO_RUNTIME_LIBS=1 PROJECT_NAME='llvm' $extra_make_flags EDIS_VERSION=1", "making libedis", 1);			
			# Combine all .o files from a bunch of static libraries from llvm
			# and clang into a single .a file.
			create_single_llvm_arhive_for_arch ($llvm_dstroot_arch, 1);
		}

		++$arch_idx;
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

sub create_single_llvm_arhive_for_arch
{
	my $arch_dstroot = shift;
    my $split_into_objects = shift;
	my @object_dirs;
	my $object_dir;
	my $tmp_dir = $arch_dstroot;
	my $arch_output_file = "$arch_dstroot/$llvm_clang_basename";
    -e $arch_output_file and return;
	my $files = "$arch_dstroot/files.txt";
	open (FILES, ">$files") or die "Can't open $! for writing...\n";

	for my $path (@archive_files) 
	{
		my $archive_fullpath = finalize_path ("$arch_dstroot/$path");
		if (-e $archive_fullpath)
		{
            if ($split_into_objects)
            {
                my ($archive_file, $archive_dir, $archive_ext) = fileparse($archive_fullpath, ('.a'));
        
                $object_dir = "$tmp_dir/$archive_file";
                push @object_dirs, $object_dir;
            
                do_command ("cd '$tmp_dir'; mkdir '$archive_file'; cd '$archive_file'; ar -x $archive_fullpath");
        
                my @objects = bsd_glob("$object_dir/*.o");
        
                foreach my $object (@objects)
                {
                    my ($o_file, $o_dir) = fileparse($object);
                    my $new_object = "$object_dir/${archive_file}-$o_file";
                    print FILES "$new_object\n";
                    do_command ("mv '$object' '$new_object'");
                }				
            }
            else
            {
                # just add the .a files into the file list
                print FILES "$archive_fullpath\n";
            }
		}
        else
        {
            print "warning: archive doesn't exist: '$archive_fullpath'\n";
        }
	}
	close (FILES);
    do_command ("libtool -static -o '$arch_output_file' -filelist '$files'");
    do_command ("ranlib '$arch_output_file'");

	foreach $object_dir (@object_dirs)
	{
		do_command ("rm -rf '$object_dir'");
	}
	do_command ("rm -rf '$files'");
}

build_llvm();
