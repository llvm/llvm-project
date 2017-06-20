#!/usr/bin/python

import os, sys, subprocess


if not "SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER" in os.environ:
  raise EnvironmentError("Specify SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER to select which simulator to use.")

device_id = os.environ["SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER"]

if "ASAN_OPTIONS" in os.environ:
    os.environ["SIMCTL_CHILD_ASAN_OPTIONS"] = os.environ["ASAN_OPTIONS"]

exitcode = subprocess.call(["xcrun", "simctl", "spawn", device_id] + sys.argv[1:])
if exitcode > 125:
  exitcode = 126
sys.exit(exitcode)
