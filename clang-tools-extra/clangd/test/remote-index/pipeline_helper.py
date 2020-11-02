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
import threading


def kill_server_after_delay(server_process):
  time.sleep(10)
  if server_process.poll() is None:
    server_process.kill()


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

  # This will kill index_server_process if it hangs without printing init
  # message.
  shutdown_thread = threading.Thread(
      target=kill_server_after_delay, args=(index_server_process,))
  shutdown_thread.daemon = True
  shutdown_thread.start()

  # Wait for the server to warm-up.
  found_init_message = False
  while index_server_process.poll() is None:
    if b'Server listening' in index_server_process.stderr.readline():
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
  index_server_process.kill()


if __name__ == '__main__':
  main()
