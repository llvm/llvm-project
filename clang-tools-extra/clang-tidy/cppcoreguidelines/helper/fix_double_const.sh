#!/bin/bash

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail

BASEDIR=$1

echo "Fixing instances of double const qualification for files in $BASEDIR..."
for i in $(find $BASEDIR -type f -iregex '.*\.\(h\|hxx\|hpp\|hh\|c\|cpp\|cxx\|cc\)$');do
    echo "Checking $i"
    sed -i 's/const const/const/g' $i
done
