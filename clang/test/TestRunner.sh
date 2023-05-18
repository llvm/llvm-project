#!/bin/sh
#
# TestRunner.sh - Backward compatible utility for testing an individual file.

# Find where this script is.
Dir=$(dirname $(which $0))
AbsDir=$(cd $Dir; pwd)

# Find 'lit', assuming standard layout.
lit=$AbsDir/../../../utils/lit/lit.py

# Dispatch to lit.
$lit "$@"
