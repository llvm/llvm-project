#!/usr/bin/env perl
#
# (C) 2016 by Argonne National Laboratory.
#     See LICENSE.txt in top-level directory.
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

    print "\n";

    exit 1;
}

sub check_package
{
    my $pack = shift;

    print "===> Checking for package $pack... ";
    if (`which $pack` eq "") {
        print "not found\n";
        exit;
    }
    print "done\n";
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
    "help" => \&usage
) or die "unable to parse options, stopped";

if (scalar(@ARGV) != 0) {
    usage();
}

if (!$branch || !$version || !$git_repo) {
    usage();
}

check_package("git");
check_package("cmake");
print("\n");


my $tdir = tempdir(CLEANUP => 1);
my $local_git_clone = "${tdir}/bolt-clone";


# clone git repo
print("===> Cloning git repo... ");
run_cmd("git clone --recursive -b ${branch} ${git_repo} ${local_git_clone}");
print("done\n");

# chdirs to $local_git_clone if valid
check_git_repo($local_git_clone);
print("\n");

if ($append_commit_id) {
    my $desc = `git describe --always ${branch}`;
    chomp $desc;
    $version .= "-${desc}";
}

my $expdir = "${tdir}/bolt-${version}";

# Clean up the log file
system("rm -f ${root}/$logfile");

# Check out the appropriate branch
print("===> Exporting code from git... ");
run_cmd("rm -rf ${expdir}");
run_cmd("mkdir -p ${expdir}");
run_cmd("git archive ${branch} --prefix='bolt-${version}/' | tar -x -C $tdir");
run_cmd("git submodule foreach --recursive \'git archive HEAD --prefix='' | tar -x -C `echo \${toplevel}/\${path} | sed -e s/clone/${version}/`'");
print("done\n");

# Remove unnecessary files
print("===> Removing unnecessary files in the main codebase... ");
chdir($expdir);
run_cmd("find . -name .gitignore | xargs rm -rf");
run_cmd("find . -name .gitmodules | xargs rm -rf");
run_cmd("find . -name .tmp | xargs rm -rf");
print("done\n");

# TODO: Get docs

# Create the main tarball
print("===> Creating the final bolt tarball... ");
chdir("${tdir}");
run_cmd("tar -czvf bolt-${version}.tar.gz bolt-${version}");
run_cmd("cp -a bolt-${version}.tar.gz ${root}/");
print("done\n");

# make sure we are outside of the tempdir so that the CLEANUP logic can run
chdir("${tdir}/..");

