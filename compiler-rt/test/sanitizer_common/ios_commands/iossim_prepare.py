#!/usr/bin/python

import json
import os
import subprocess


device_id = os.environ.get('SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER')
if not device_id:
    raise EnvironmentError('Specify SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER to select which simulator to use.')

boot_cmd = ['xcrun', 'simctl', 'bootstatus', device_id, '-b']
subprocess.check_call(boot_cmd)
# TODO(rdar58118442): we start the simulator here, but we never tear it down


print(json.dumps({"env": {}}))
