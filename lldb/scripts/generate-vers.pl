#!/usr/bin/perl

sub usage()
{
  print "Usage: generate-vers.pl /path/toproject.pbxproj";
  exit(0);
}

(scalar @ARGV == 1) or usage();

open $pbxproj, $ARGV[0] or die "Couldn't open ".$ARGV[0];

$current_project_version = None;
$product_name = None;

while ($line = <$pbxproj>)
{
  chomp ($line);
  
  if ($current_project_version == None &&
      $line =~ /CURRENT_PROJECT_VERSION = ([0-9]+)/)
  {
    $current_project_version = $1;
  }
  
  if ($product_name == None &&
      $line =~ /productName = ([^;]+)/)
  {
    $product_name = $1;
  }
}

if (!$product_name || !$current_project_version)
{
  print "Couldn't get needed information from the .pbxproj";
  exit(-1);
}

$uppercase_name = uc $product_name;
$lowercase_name = lc $product_name;

close $pbxproj;

$file_string = " const unsigned char liblldb_coreVersionString[] __attribute__ ((used)) = \"@(#)PROGRAM:".$uppercase_name."  PROJECT:".$lowercase_name."-".$current_project_version."\" \"\\n\"; const double liblldb_coreVersionNumber __attribute__ ((used)) = (double)".$current_project_version.".;\n";

print $file_string;
