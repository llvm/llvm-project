#!/bin/sh
ccache "$@"
"$@" --analyze --analyzer-output html -o analyzer-results \
     -Xclang -analyzer-config -Xclang max-nodes=75000
