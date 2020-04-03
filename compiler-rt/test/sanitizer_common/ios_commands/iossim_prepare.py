#!/usr/bin/python

import json
import os
import subprocess


device_id = os.environ.get('SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER')
if not device_id:
    raise EnvironmentError('Specify SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER to select which simulator to use.')

DEVNULL = open(os.devnull, 'w')
subprocess.call(['xcrun', 'simctl', 'shutdown', device_id], stderr=DEVNULL)
subprocess.check_call(['xcrun', 'simctl', 'boot', device_id])
# TODO(rdar58118442): we start the simulator here, but we never tear it down


print(json.dumps({"env": {}}))
