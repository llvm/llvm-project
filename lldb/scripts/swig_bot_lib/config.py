#!/usr/bin/env python

"""
Shared functionality used by `client` and `server` when dealing with
configuration data
"""

# Future imports
from __future__ import absolute_import
from __future__ import print_function

# Python modules
import json
import logging
import os
import socket
import struct
import sys

# LLDB modules
import use_lldb_suite

# package imports
from . import local

def generate_config_json(options):
    config = {"languages": options.languages}
    return json.dumps(config)

def parse_config_json(option_json):
    return json.loads(option_json)
