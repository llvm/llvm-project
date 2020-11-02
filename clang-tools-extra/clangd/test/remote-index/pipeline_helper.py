#!/usr/bin/env python
#
#===- pipeline_helper.py - Remote Index pipeline Helper *- python -------*--===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

import argparse
import os
import subprocess
from socket import socket
import sys
import time
import signal


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-file-name', required=True)
  parser.add_argument('--project-root', required=True)
  parser.add_argument('--index-file', required=True)

  args = parser.parse_args()

  # Grab an available port.
  with socket() as s:
    s.bind(('localhost', 0))
    server_address = 'localhost:' + str(s.getsockname()[1])

  index_server_process = subprocess.Popen([
      'clangd-index-server', '--server-address=' + server_address,
      args.index_file, args.project_root
  ],
                                          stderr=subprocess.PIPE)

  # Wait for the server to warm-up.
  time.sleep(4)
  found_init_message = False
  for line in index_server_process.stderr:
    if b'Server listening' in line:
      found_init_message = True
      break

  if not found_init_message:
    sys.exit(1)

  in_file = open(args.input_file_name)

  clangd_process = subprocess.Popen([
      'clangd', '--remote-index-address=' + server_address,
      '--project-root=' + args.project_root, '--lit-test', '--sync'
  ],
                                    stdin=in_file)

  clangd_process.wait()
  os.kill(index_server_process.pid, signal.SIGINT)


if __name__ == '__main__':
  main()
