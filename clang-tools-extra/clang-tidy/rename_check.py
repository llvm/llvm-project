#!/usr/bin/env python
#
#===- rename_check.py - clang-tidy check renamer ------------*- python -*--===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===-----------------------------------------------------------------------===#

from __future__ import unicode_literals

import argparse
import glob
import io
import os
import re

def replaceInFileRegex(fileName, sFrom, sTo):
  if sFrom == sTo:
    return

  # The documentation files are encoded using UTF-8, however on Windows the
  # default encoding might be different (e.g. CP-1252). To make sure UTF-8 is
  # always used, use `io.open(filename, mode, encoding='utf8')` for reading and
  # writing files here and elsewhere.
  txt = None
  with io.open(fileName, 'r', encoding='utf8') as f:
    txt = f.read()

  txt = re.sub(sFrom, sTo, txt)
  print("Replacing '%s' -> '%s' in '%s'..." % (sFrom, sTo, fileName))
  with io.open(fileName, 'w', encoding='utf8') as f:
    f.write(txt)


def replaceInFile(fileName, sFrom, sTo):
  if sFrom == sTo:
    return
  txt = None
  with io.open(fileName, 'r', encoding='utf8') as f:
    txt = f.read()

  if sFrom not in txt:
    return

  txt = txt.replace(sFrom, sTo)
  print("Replacing '%s' -> '%s' in '%s'..." % (sFrom, sTo, fileName))
  with io.open(fileName, 'w', encoding='utf8') as f:
    f.write(txt)


def generateCommentLineHeader(filename):
  return ''.join(['//===--- ',
                  os.path.basename(filename),
                  ' - clang-tidy ',
                  '-' * max(0, 42 - len(os.path.basename(filename))),
                  '*- C++ -*-===//'])


def generateCommentLineSource(filename):
  return ''.join(['//===--- ',
                  os.path.basename(filename),
                  ' - clang-tidy',
                  '-' * max(0, 52 - len(os.path.basename(filename))),
                  '-===//'])


def fileRename(fileName, sFrom, sTo):
  if sFrom not in fileName or sFrom == sTo:
    return fileName
  newFileName = fileName.replace(sFrom, sTo)
  print("Renaming '%s' -> '%s'..." % (fileName, newFileName))
  os.rename(fileName, newFileName)
  return newFileName


def deleteMatchingLines(fileName, pattern):
  lines = None
  with io.open(fileName, 'r', encoding='utf8') as f:
    lines = f.readlines()

  not_matching_lines = [l for l in lines if not re.search(pattern, l)]
  if len(not_matching_lines) == len(lines):
    return False

  print("Removing lines matching '%s' in '%s'..." % (pattern, fileName))
  print('  ' + '  '.join([l for l in lines if re.search(pattern, l)]))
  with io.open(fileName, 'w', encoding='utf8') as f:
    f.writelines(not_matching_lines)

  return True


def getListOfFiles(clang_tidy_path):
  files = glob.glob(os.path.join(clang_tidy_path, '*'))
  for dirname in files:
    if os.path.isdir(dirname):
      files += glob.glob(os.path.join(dirname, '*'))
  files += glob.glob(os.path.join(clang_tidy_path, '..', 'test',
                                  'clang-tidy', '*'))
  files += glob.glob(os.path.join(clang_tidy_path, '..', 'docs',
                                  'clang-tidy', 'checks', '*'))
  return [filename for filename in files if os.path.isfile(filename)]


# Adapts the module's CMakelist file. Returns 'True' if it could add a new
# entry and 'False' if the entry already existed.
def adapt_cmake(module_path, check_name_camel):
  filename = os.path.join(module_path, 'CMakeLists.txt')
  with io.open(filename, 'r', encoding='utf8') as f:
    lines = f.readlines()

  cpp_file = check_name_camel + '.cpp'

  # Figure out whether this check already exists.
  for line in lines:
    if line.strip() == cpp_file:
      return False

  print('Updating %s...' % filename)
  with io.open(filename, 'w', encoding='utf8') as f:
    cpp_found = False
    file_added = False
    for line in lines:
      cpp_line = line.strip().endswith('.cpp')
      if (not file_added) and (cpp_line or cpp_found):
        cpp_found = True
        if (line.strip() > cpp_file) or (not cpp_line):
          f.write('  ' + cpp_file + '\n')
          file_added = True
      f.write(line)

  return True

# Modifies the module to include the new check.
def adapt_module(module_path, module, check_name, check_name_camel):
  modulecpp = next(iter(filter(
      lambda p: p.lower() == module.lower() + 'tidymodule.cpp',
      os.listdir(module_path))))
  filename = os.path.join(module_path, modulecpp)
  with io.open(filename, 'r', encoding='utf8') as f:
    lines = f.readlines()

  print('Updating %s...' % filename)
  with io.open(filename, 'w', encoding='utf8') as f:
    header_added = False
    header_found = False
    check_added = False
    check_decl = ('    CheckFactories.registerCheck<' + check_name_camel +
                  '>(\n        "' + check_name + '");\n')

    for line in lines:
      if not header_added:
        match = re.search('#include "(.*)"', line)
        if match:
          header_found = True
          if match.group(1) > check_name_camel:
            header_added = True
            f.write('#include "' + check_name_camel + '.h"\n')
        elif header_found:
          header_added = True
          f.write('#include "' + check_name_camel + '.h"\n')

      if not check_added:
        if line.strip() == '}':
          check_added = True
          f.write(check_decl)
        else:
          match = re.search('registerCheck<(.*)>', line)
          if match and match.group(1) > check_name_camel:
            check_added = True
            f.write(check_decl)
      f.write(line)


# Adds a release notes entry.
def add_release_notes(clang_tidy_path, old_check_name, new_check_name):
  filename = os.path.normpath(os.path.join(clang_tidy_path,
                                           '../docs/ReleaseNotes.rst'))
  with io.open(filename, 'r', encoding='utf8') as f:
    lines = f.readlines()

  lineMatcher = re.compile('Renamed checks')
  nextSectionMatcher = re.compile('Improvements to include-fixer')
  checkMatcher = re.compile('- The \'(.*)')

  print('Updating %s...' % filename)
  with io.open(filename, 'w', encoding='utf8') as f:
    note_added = False
    header_found = False
    add_note_here = False

    for line in lines:
      if not note_added:
        match = lineMatcher.match(line)
        match_next = nextSectionMatcher.match(line)
        match_check = checkMatcher.match(line)
        if match_check:
          last_check = match_check.group(1)
          if last_check > old_check_name:
            add_note_here = True

        if match_next:
          add_note_here = True

        if match:
          header_found = True
          f.write(line)
          continue

        if line.startswith('^^^^'):
          f.write(line)
          continue

        if header_found and add_note_here:
          if not line.startswith('^^^^'):
            f.write("""- The '%s' check was renamed to :doc:`%s
  <clang-tidy/checks/%s>`

""" % (old_check_name, new_check_name, new_check_name))
            note_added = True

      f.write(line)

def main():
  parser = argparse.ArgumentParser(description='Rename clang-tidy check.')
  parser.add_argument('old_check_name', type=str,
                      help='Old check name.')
  parser.add_argument('new_check_name', type=str,
                      help='New check name.')
  parser.add_argument('--check_class_name', type=str,
                      help='Old name of the class implementing the check.')
  args = parser.parse_args()

  old_module = args.old_check_name.split('-')[0]
  new_module = args.new_check_name.split('-')[0]
  if args.check_class_name:
    check_name_camel = args.check_class_name
  else:
    check_name_camel = (''.join(map(lambda elem: elem.capitalize(),
                                    args.old_check_name.split('-')[1:])) +
                        'Check')

  new_check_name_camel = (''.join(map(lambda elem: elem.capitalize(),
                                      args.new_check_name.split('-')[1:])) +
                          'Check')

  clang_tidy_path = os.path.dirname(__file__)

  header_guard_variants = [
      (args.old_check_name.replace('-', '_')).upper() + '_CHECK',
      (old_module + '_' + check_name_camel).upper(),
      (old_module + '_' + new_check_name_camel).upper(),
      args.old_check_name.replace('-', '_').upper()]
  header_guard_new = (new_module + '_' + new_check_name_camel).upper()

  old_module_path = os.path.join(clang_tidy_path, old_module)
  new_module_path = os.path.join(clang_tidy_path, new_module)

  if (args.old_check_name != args.new_check_name):
    # Remove the check from the old module.
    cmake_lists = os.path.join(old_module_path, 'CMakeLists.txt')
    check_found = deleteMatchingLines(cmake_lists, '\\b' + check_name_camel)
    if not check_found:
      print("Check name '%s' not found in %s. Exiting." %
            (check_name_camel, cmake_lists))
      return 1

    modulecpp = next(iter(filter(
        lambda p: p.lower() == old_module.lower() + 'tidymodule.cpp',
        os.listdir(old_module_path))))
    deleteMatchingLines(os.path.join(old_module_path, modulecpp),
                      '\\b' + check_name_camel + '|\\b' + args.old_check_name)

  for filename in getListOfFiles(clang_tidy_path):
    originalName = filename
    filename = fileRename(filename, args.old_check_name,
                          args.new_check_name)
    filename = fileRename(filename, check_name_camel, new_check_name_camel)
    replaceInFile(filename, generateCommentLineHeader(originalName),
                  generateCommentLineHeader(filename))
    replaceInFile(filename, generateCommentLineSource(originalName),
                  generateCommentLineSource(filename))
    for header_guard in header_guard_variants:
      replaceInFile(filename, header_guard, header_guard_new)

    if args.new_check_name + '.rst' in filename:
      replaceInFile(
          filename,
          args.old_check_name + '\n' + '=' * len(args.old_check_name) + '\n',
          args.new_check_name + '\n' + '=' * len(args.new_check_name) + '\n')

    replaceInFile(filename, args.old_check_name, args.new_check_name)
    replaceInFile(filename, old_module + '::' + check_name_camel,
                  new_module + '::' + new_check_name_camel)
    replaceInFile(filename, old_module + '/' + check_name_camel,
                  new_module + '/' + new_check_name_camel)
    replaceInFile(filename, check_name_camel, new_check_name_camel)

  if old_module != new_module or new_module == 'llvm':
    if new_module == 'llvm':
      new_namespace = new_module + '_check'
    else:
      new_namespace = new_module
    check_implementation_files = glob.glob(
        os.path.join(old_module_path, new_check_name_camel + '*'))
    for filename in check_implementation_files:
      # Move check implementation to the directory of the new module.
      filename = fileRename(filename, old_module_path, new_module_path)
      replaceInFileRegex(filename, 'namespace ' + old_module + '[^ \n]*',
                         'namespace ' + new_namespace)

  if (args.old_check_name == args.new_check_name):
    return

  # Add check to the new module.
  adapt_cmake(new_module_path, new_check_name_camel)
  adapt_module(new_module_path, new_module, args.new_check_name,
               new_check_name_camel)

  os.system(os.path.join(clang_tidy_path, 'add_new_check.py')
            + ' --update-docs')
  add_release_notes(clang_tidy_path, args.old_check_name, args.new_check_name)


if __name__ == '__main__':
  main()
