#! /bin/bash

# This script is intended as a FileCheck replacement to update the test
# expectations in a -ast-dump test.
#
# Usage:
#
# $ lit -DFileCheck=$PWD/utils/make-ast-dump-check.sh test/AST/ast-dump-openmp-*

prefix=CHECK

while [[ "$#" -ne 0 ]]; do
  case "$1" in
  --check-prefix)
    shift
    prefix="$1"
    ;;
  --implicit-check-not)
    shift
    ;;
  -*)
    ;;
  *)
    file="$1"
    ;;
  esac
  shift
done

testdir="$(dirname "$file")"

read -r -d '' script <<REWRITE
BEGIN {
  skipping_builtins = 0
  matched_last_line = 0
}

/^[\`|].* line:/ {
  skipping_builtins = 0
}

{
  if (skipping_builtins == 1) {
    matched_last_line = 0
    next
  }
}

/TranslationUnitDecl/ {
  skipping_builtins = 1
}

{
  s = \$0
  gsub("0x[0-9a-fA-F]+", "{{.*}}", s)
  gsub("$testdir/", "{{.*}}", s)
}

matched_last_line == 0 {
  print "// ${prefix}: " s
}

matched_last_line == 1 {
  print "// ${prefix}-NEXT: " s
}

{
  matched_last_line = 1
}
REWRITE

echo "$script"

{
  cat "$file" | grep -v "$prefix"
  awk "$script"
} > "$file.new"

mv "$file.new" "$file"
