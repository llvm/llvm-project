#!/usr/bin/perl

sub usage()
{
  print "Usage: generate-vers.pl /path/toproject.pbxproj";
  exit(0);
}

(scalar @ARGV == 1) or usage();

open $pbxproj, $ARGV[0] or die "Couldn't open ".$ARGV[0];

$lldb_version = None;
$lldb_train = None;
$lldb_revision = None;
$lldb_version_string = None;

$product_name = "lldb";

while ($line = <$pbxproj>)
{
  chomp ($line);
  
  if ($lldb_version == None &&
      $line =~ /CURRENT_PROJECT_VERSION = ([0-9]+).([0-9]+).([0-9]+)(.[0-9])?/)
  {
    $lldb_version = $1;
    $lldb_train = $2;
    $lldb_revision = $3;
    $lldb_patchlevel = $4;

    if ($lldb_patchlevel != None)
    {
      $lldb_version_string = $lldb_version.".".$lldb_train.".".$lldb_revision.".".$lldb_patchlevel;
    }
    else
    {
      $lldb_version_string = $lldb_version.".".$lldb_train.".".$lldb_revision;
    } 
  }
}

if (!$product_name || !$lldb_version_string)
{
  print "Couldn't get needed information from the .pbxproj";
  exit(-1);
}

$uppercase_name = uc $product_name;
$lowercase_name = lc $product_name;

close $pbxproj;

$file_string = " const unsigned char liblldb_coreVersionString[] __attribute__ ((used)) = \"@(#)PROGRAM:".$uppercase_name."  PROJECT:".$lowercase_name."-".$lldb_version_string."\" \"\\n\"; const double liblldb_coreVersionNumber __attribute__ ((used)) = (double)".$lldb_version.".".$lldb_train.";\n";

print $file_string;
