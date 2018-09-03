#!/usr/bin/python

import os, sys, subprocess


if not "SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER" in os.environ:
  raise EnvironmentError("Specify SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER to select which simulator to use.")

device_id = os.environ["SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER"]

for e in ["ASAN_OPTIONS", "TSAN_OPTIONS", "UBSAN_OPTIONS"]:
  if e in os.environ:
    os.environ["SIMCTL_CHILD_" + e] = os.environ[e]

exitcode = subprocess.call(["xcrun", "simctl", "spawn", device_id] + sys.argv[1:])
if exitcode > 125:
  exitcode = 126
sys.exit(exitcode)
