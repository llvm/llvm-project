#!/usr/bin/env python

import glob, os, pipes, sys, subprocess


device_id = os.environ.get('SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER')
if not device_id:
  raise EnvironmentError("Specify SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER to select which simulator to use.")

for e in ["ASAN_OPTIONS", "TSAN_OPTIONS", "UBSAN_OPTIONS", "APPLE_ASAN_INIT_FOR_DLOPEN", "ASAN_ACTIVATION_OPTIONS"]:
  if e in os.environ:
    os.environ["SIMCTL_CHILD_" + e] = os.environ[e]

prog = sys.argv[1]
exit_code = None
if prog == 'rm':
  # The simulator and host actually share the same file system so we can just
  # execute directly on the host.
  rm_args = []
  for arg in sys.argv[2:]:
    if '*' in arg or '?' in arg:
      # Don't quote glob pattern
      rm_args.append(arg)
    else:
      # FIXME(dliew): pipes.quote() is deprecated
      rm_args.append(pipes.quote(arg))
  rm_cmd_line = ["/bin/rm"] + rm_args
  rm_cmd_line_str = ' '.join(rm_cmd_line)
  # We use `shell=True` so that any wildcard globs get expanded by the shell.
  exitcode = subprocess.call(rm_cmd_line_str, shell=True)
else:
  exitcode = subprocess.call(["xcrun", "simctl", "spawn", "--standalone", device_id] + sys.argv[1:])
if exitcode > 125:
  exitcode = 126
sys.exit(exitcode)
