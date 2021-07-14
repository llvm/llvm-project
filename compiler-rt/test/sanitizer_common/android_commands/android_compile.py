#!/usr/bin/env python

import os, sys, subprocess
from android_common import *


here = os.path.abspath(os.path.dirname(sys.argv[0]))
android_run = os.path.join(here, 'android_run.py')

output = None
output_type = 'executable'

args = sys.argv[1:]
while args:
    arg = args.pop(0)
    if arg == '-shared':
        output_type = 'shared'
    elif arg == '-c':
        output_type = 'object'
    elif arg == '-o':
        output = args.pop(0)

if output == None:
    print("No output file name!")
    sys.exit(1)

ret = subprocess.call(sys.argv[1:])
if ret != 0:
    sys.exit(ret)

if output_type in ['executable', 'shared']:
    push_to_device(output)

if output_type == 'executable':
    os.rename(output, output + '.real')
    os.symlink(android_run, output)
