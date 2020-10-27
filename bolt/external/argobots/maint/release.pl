#!/usr/bin/env perl
#
# See COPYRIGHT in top-level directory.
#

use strict;
use warnings;

use Cwd qw( cwd getcwd realpath );
use Getopt::Long;
use File::Temp qw( tempdir );

my $arg = 0;
my $branch = "";
my $version = "";
my $append_commit_id;
my $root = cwd();
my $with_autoconf = "";
my $with_automake = "";
my $git_repo = "";

my $logfile = "release.log";

sub usage
{
    print "Usage: $0 [OPTIONS]\n\n";
    print "OPTIONS:\n";

    print "\t--git-repo           path to root of the git repository (required)\n";
    print "\t--branch             git branch to be packaged (required)\n";
    print "\t--version            tarball version (required)\n";
    print "\t--append-commit-id   append git commit description (optional)\n";
    print "\t--with-autoconf      autoconf directory (optional)\n";
    print "\t--with-automake      automake directory (optional)\n";

    print "\n";

    exit 1;
}

sub check_package
{
    my $pack = shift;

    print "===> Checking for package $pack... ";
    if ($with_autoconf and ($pack eq "autoconf")) {
        # the user specified dir where autoconf can be found
        if (not -x "$with_autoconf/$pack") {
            print "not found\n";
            exit;
        }
    }
    if ($with_automake and ($pack eq "automake")) {
        # the user specified dir where automake can be found
        if (not -x "$with_automake/$pack") {
            print "not found\n";
            exit;
        }
    }
    else {
        if (`which $pack` eq "") {
            print "not found\n";
            exit;
        }
    }
    print "done\n";
}

sub check_autotools_version
{
    my $tool = shift;
    my $req_ver = shift;
    my $curr_ver;

    $curr_ver = `$tool --version | head -1 | cut -f4 -d' ' | xargs echo -n`;
    if ("$curr_ver" ne "$req_ver") {
	print("\tERROR: $tool version mismatch ($req_ver) required\n\n");
	exit;
    }
}

# will also chdir to the top level of the git repository
sub check_git_repo {
    my $repo_path = shift;

    print "===> chdir to $repo_path\n";
    chdir $repo_path;

    print "===> Checking git repository sanity... ";
    unless (`git rev-parse --is-inside-work-tree 2> /dev/null` eq "true\n") {
        print "ERROR: $repo_path is not a git repository\n";
        exit 1;
    }
    # I'm not strictly sure that this is true, but it's not too burdensome right
    # now to restrict it to complete (non-bare repositories).
    unless (`git rev-parse --is-bare-repository 2> /dev/null` eq "false\n") {
        print "ERROR: $repo_path is a *bare* repository (need working tree)\n";
        exit 1;
    }

    print "done\n";
}


sub run_cmd
{
    my $cmd = shift;

    #print("===> running cmd=|$cmd| from ".getcwd()."\n");
    system("$cmd >> $root/$logfile 2>&1");
    if ($?) {
        die "unable to execute ($cmd), \$?=$?.  Stopped";
    }
}

GetOptions(
    "git-repo=s" => \$git_repo,
    "branch=s" => \$branch,
    "version=s" => \$version,
    "append-commit-id!" => \$append_commit_id,
    "with-autoconf=s" => \$with_autoconf,
    "with-automake=s" => \$with_automake,
    "help"     => \&usage
) or die "unable to parse options, stopped";

if (scalar(@ARGV) != 0) {
    usage();
}

if (!$branch || !$version || !$git_repo) {
    usage();
}

check_package("git");
check_package("latex");
check_package("autoconf");
check_package("automake");
print("\n");


my $tdir = tempdir(CLEANUP => 1);
my $local_git_clone = "${tdir}/argobots-clone";


# clone git repo
print("===> Cloning git repo... ");
run_cmd("git clone ${git_repo} ${local_git_clone}");
print("done\n");

# chdirs to $local_git_clone if valid
check_git_repo($local_git_clone);
print("\n");

my $current_ver = `git show ${branch}:maint/version.m4 | grep ABT_VERSION_m4 | \
                   sed -e 's/^.*\\[ABT_VERSION_m4\\],\\[\\(.*\\)\\].*/\\1/g'`;
if ("$current_ver" ne "$version\n") {
    print("\tWARNING: maint/version does not match user version\n\n");
}

my $changes_ver = `git show ${branch}:CHANGES | grep "http://git.mcs.anl.gov/argo/argobots.git/shortlog" | \
                   sed -e '2,\$d' -e 's/.*\.\.//g'`;
if ("$changes_ver" ne "$version\n") {
    print("\tWARNING: CHANGES/version does not match user version\n\n");
}

if ($append_commit_id) {
    my $desc = `git describe --always ${branch}`;
    chomp $desc;
    $version .= "-${desc}";
}

my $expdir = "${tdir}/argobots-${version}";

# Clean up the log file
system("rm -f ${root}/$logfile");

# Check out the appropriate branch
print("===> Exporting code from git... ");
run_cmd("rm -rf ${expdir}");
run_cmd("mkdir -p ${expdir}");
run_cmd("git archive ${branch} --prefix='argobots-${version}/' | tar -x -C $tdir");
print("done\n");

print("===> Create release date and version information... ");
chdir($expdir);

my $date = `date`;
chomp $date;
system(qq(perl -p -i -e 's/\\[ABT_RELEASE_DATE_m4\\],\\[unreleased development copy\\]/[ABT_RELEASE_DATE_m4],[$date]/g' ./maint/version.m4));
print("done\n");

# Create configure
print("===> Creating configure in the main codebase... ");
chdir($expdir);
{
    my $cmd = "./autogen.sh";
    $cmd .= " --with-autoconf=$with_autoconf" if $with_autoconf;
    $cmd .= " --with-automake=$with_automake" if $with_automake;
    run_cmd($cmd);
}
print("done\n");

# Remove unnecessary files
print("===> Removing unnecessary files in the main codebase... ");
chdir($expdir);
run_cmd("rm -rf README.md");
run_cmd("find . -name autom4te.cache | xargs rm -rf");
run_cmd("find . -name .gitignore | xargs rm -rf");
run_cmd("find . -name .github | xargs rm -rf");
run_cmd("find . -name .tmp | xargs rm -rf");
print("done\n");

# TODO: Get docs
#print("===> Creating secondary codebase for the docs... ");
#run_cmd("mkdir ${expdir}-build");
#chdir("${expdir}-build");
#run_cmd("${expdir}/configure");
#run_cmd("(make doxygen)");
#print("done\n");
#
#print("===> Copying docs over... ");
#run_cmd("cp -a doc/doxygen/html ${expdir}/");
#print("done\n");

# Create the main tarball
print("===> Creating the final argobots tarball... ");
chdir("${tdir}");
run_cmd("tar -czvf argobots-${version}.tar.gz argobots-${version}");
run_cmd("cp -a argobots-${version}.tar.gz ${root}/");
print("done\n");

# make sure we are outside of the tempdir so that the CLEANUP logic can run
chdir("${tdir}/..");
