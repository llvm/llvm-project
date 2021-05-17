#!/usr/bin/env python

# Converts clang-scan-deps output into response files.
#   * For modules, arguments in the resulting response file are enough to build a PCM.
#   * For translation units, the response file needs to be added to the original Clang invocation from compilation
#     database.
#
# Usage:
#
#   clang-scan-deps -compilation-database compile_commands.json ... > deps.json
#   module-deps-to-rsp.py deps.json --module-name=ModuleName > module_name.cc1.rsp
#   module-deps-to-rsp.py deps.json --tu-index=0 > tu.rsp
#   clang @module_name.cc1.rsp
#   clang ... @tu.rsp

import argparse
import json
import sys

class ModuleNotFoundError(Exception):
  def __init__(self, module_name):
    self.module_name = module_name

class FullDeps:
  def __init__(self):
    self.modules = dict()
    self.translation_units = str()
    
def getModulePathArgs(modules, full_deps):
  cmd = []
  for md in modules:
    m = full_deps.modules[md['module-name'] + '-' + md['context-hash']]
    cmd += [u'-fmodule-map-file=' + m['clang-modulemap-file']]
    cmd += [u'-fmodule-file=' + md['module-name'] + '-' + md['context-hash'] + '.pcm']
  return cmd

def getCommandLineForModule(module_name, full_deps):
  for m in full_deps.modules.values():
    if m['name'] == module_name:
      module = m
      break
  else:
    raise ModuleNotFoundError(module_name)

  cmd = m['command-line']
  cmd += getModulePathArgs(m['clang-module-deps'], full_deps)
  cmd += [u'-o', m['name'] + '-' + m['context-hash'] + '.pcm']
  cmd += [m['clang-modulemap-file']]
  
  return cmd
  
def getCommandLineForTU(tu, full_deps):
  cmd = tu['command-line']
  cmd += getModulePathArgs(tu['clang-module-deps'], full_deps)
  return cmd

def parseFullDeps(json):
  ret = FullDeps()
  for m in json['modules']:
    ret.modules[m['name'] + '-' + m['context-hash']] = m
  ret.translation_units = json['translation-units']
  return ret

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("full_deps_file", help="Path to the full dependencies json file",
                      type=str)
  action = parser.add_mutually_exclusive_group(required=True)
  action.add_argument("--module-name", help="The name of the module to get arguments for",
                      type=str)
  action.add_argument("--tu-index", help="The index of the translation unit to get arguments for",
                      type=int)
  args = parser.parse_args()
  
  full_deps = parseFullDeps(json.load(open(args.full_deps_file, 'r')))
  
  try:
    if args.module_name:
      print(" ".join(getCommandLineForModule(args.module_name, full_deps)))
    
    elif args.tu_index != None:
      print(" ".join(getCommandLineForTU(full_deps.translation_units[args.tu_index], full_deps)))
  except:
    print("Unexpected error:", sys.exc_info()[0])
    raise

if __name__ == '__main__':
  main()
