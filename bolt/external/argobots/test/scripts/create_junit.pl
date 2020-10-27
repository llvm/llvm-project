#!/usr/bin/env perl
# -*- Mode: perl; -*-
#
# See COPYRIGHT in top-level directory.
#

use strict;
use warnings;

use Getopt::Long;

my $test_dir = "";
my $log_filename = "";
my $list_filename = "";
my $junit_filename = "summary.junit.xml";

sub usage
{
    print "Usage: $0 [OPTIONS]\n\n";
    print "OPTIONS:\n";
    print "\t--test-dir        test directory (required)\n";
    print "\t--log-filename    log filename of test suite (required)\n";
    print "\t--list-filename   test list filename (required)\n";
    print "\t--junit-filename  junit filename (optional)\n";
    print "\n";
    exit 1;
}

GetOptions(
    "test-dir=s" => \$test_dir,
    "log-filename=s" => \$log_filename,
    "list-filename=s" => \$list_filename,
    "junit-filename=s" => \$junit_filename,
    "help" => \&usage
) or die "unable to parse options, stopped";

if (scalar(@ARGV) != 0) {
    usage();
}

if (!$test_dir || !$log_filename || !$list_filename) {
    usage();
}

my $total_run = 0;
my $skip_count = 0;
my $xfail_count = 0;
my $fail_count = 0;
my $err_count = 0;
my $date = `date "+%Y-%m-%d-%H-%M"`;
$date =~ s/\r?\n//;
my $name = $log_filename =~ s/\./_/gr;

# parse the test results
open(LOG, "<$test_dir/$log_filename") || die "cannot open $test_dir/$log_filename";
while ($_ = <LOG>) {
    if ($_ =~ /(argobots.*)\/$log_filename/) {
        $name = $1;
    }
    elsif ($_ =~ /^# TOTAL.*/) {
        my @fields = split(/\s+/, $_);
        $total_run = $fields[2];
    }
    elsif ($_ =~ /^# SKIP.*/) {
        my @fields = split(/\s+/, $_);
        $skip_count = $fields[2];
    }
    elsif ($_ =~ /^# XFAIL.*/) {
        my @fields = split(/\s+/, $_);
        $xfail_count = $fields[2];
    }
    elsif ($_ =~ /^# FAIL.*/) {
        my @fields = split(/\s+/, $_);
        $fail_count = $fields[2];
    }
    elsif ($_ =~ /^# ERROR.*/) {
        my @fields = split(/\s+/, $_);
        $err_count = $fields[2];
    }
}
$skip_count += $xfail_count;
close(LOG);

# JUnit output file
open(JUNITOUT, ">$junit_filename") || die "cannot open $junit_filename";
print JUNITOUT "<testsuites>\n";
print JUNITOUT "  <testsuite name=\"$name\"\n";
print JUNITOUT "             tests=\"$total_run\"\n";
print JUNITOUT "             failures=\"$fail_count\"\n";
print JUNITOUT "             errors=\"$err_count\"\n";
print JUNITOUT "             skipped=\"$skip_count\"\n";
print JUNITOUT "             timestamp=\"$date\">\n";

# Check each test result and embed its log to the JUnit output file
my $test_num = 1;
open(LIST, "<$test_dir/$list_filename") || die "cannot open $test_dir/$list_filename";
while ($_ = <LIST>) {
    if ($_ =~ /^(PASS|SKIP|XFAIL|FAIL|XPASS|ERROR): (.*)$/) {
        my $result = $1;
        my $testname = $2;
        my $end_tag = "";
        print JUNITOUT "    <testcase name=\"$test_num - ./$testname\">\n";

        if ($result =~ /X?PASS/) {
            print JUNITOUT "      <system-out><![CDATA[";
            $end_tag = "]]></system-out>";
        }
        elsif ($result eq "SKIP") {
            print JUNITOUT "      <skipped type=\"TestSkipped\" ";
            print JUNITOUT "message=\"test skipped\"><![CDATA[";
            $end_tag = "]]></skipped>";
        }
        elsif ($result eq "XFAIL") {
            print JUNITOUT "      <skipped type=\"TestXfailed\" ";
            print JUNITOUT "message=\"xfail tests disabled\"><![CDATA[";
            $end_tag = "]]></skipped>";
        }
        elsif ($result eq "FAIL") {
            print JUNITOUT "      <failure type=\"TestFailed\" ";
            print JUNITOUT "message=\"failed $test_num - ./$testname\"><![CDATA[";
            $end_tag = "]]></failure>";
        }
        elsif ($result eq "ERROR") {
            print JUNITOUT "      <error type=\"TestError\" ";
            print JUNITOUT "message=\"error $test_num - ./$testname\"><![CDATA[";
            $end_tag = "]]></error>";
        }
        else {
            print "SHOULD NOT REACH HERE!!\n";
            exit 1;
        }

        my $test_log = "$test_dir/$testname.log";
        open(TLOG, "<$test_log") || die "cannot open $test_log";
        while ($_ = <TLOG>) {
            print JUNITOUT $_;
        }
        close(TLOG);

        print JUNITOUT "      $end_tag\n";
        print JUNITOUT "    </testcase>\n";
        $test_num++;
    }
}
close(LIST);

print JUNITOUT "    <system-out></system-out>\n";
print JUNITOUT "    <system-err></system-err>\n";
print JUNITOUT "  </testsuite>\n";
print JUNITOUT "</testsuites>\n";
close(JUNITOUT);

