#!/usr/bin/env python

import os, sys, subprocess

output = None
output_type = 'executable'

args = sys.argv[1:]
while args:
    arg = args.pop(0)
    if arg == '-shared':
        output_type = 'shared'
    elif arg == '-dynamiclib':
        output_type = 'dylib'
    elif arg == '-c':
        output_type = 'object'
    elif arg == '-S':
        output_type = 'assembly'
    elif arg == '-o':
        output = args.pop(0)

if output == None:
    print "No output file name!"
    sys.exit(1)

ret = subprocess.call(sys.argv[1:])
if ret != 0:
    sys.exit(ret)

# If we produce a dylib, ad-hoc sign it.
if output_type in ['shared', 'dylib']:
    ret = subprocess.call(["codesign", "-s", "-", output])
