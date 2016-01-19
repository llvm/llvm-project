#!/usr/bin/env python

r"""
Create stubs for check documentation files.
"""

import os
import re
import sys

def main():
  clang_tidy_dir = os.path.normpath(
      os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..',
                   'clang-tidy'))

  checks_doc_dir = os.path.normpath(
      os.path.join(clang_tidy_dir, '..', 'docs', 'clang-tidy', 'checks'))

  registered_checks = {}
  defined_checks = {}

  for dir_name, subdir_list, file_list in os.walk(clang_tidy_dir):
    print('Processing directory ' + dir_name + '...')
    for file_name in file_list:
      full_name = os.path.join(dir_name, file_name)
      if file_name.endswith('Module.cpp'):
        print('Module ' + file_name)
        with open(full_name, 'r') as f:
          text = f.read()
        for class_name, check_name in re.findall(
            r'\.\s*registerCheck\s*<\s*([A-Za-z0-9:]+)\s*>\(\s*"([a-z0-9-]+)"',
            text):
          registered_checks[check_name] = class_name
      elif file_name.endswith('.h'):
        print('    ' + file_name + '...')
        with open(full_name, 'r') as f:
          text = f.read()
        for comment, _, _, class_name in re.findall(
            r'((([\r\n]//)[^\r\n]*)*)\s+class (\w+)\s*:' +
            '\s*public\s+ClangTidyCheck\s*\{', text):
          defined_checks[class_name] = comment

  print('Registered checks [%s]: [%s]' %
        (len(registered_checks), registered_checks))
  print('Check implementations: %s' % len(defined_checks))

  checks = registered_checks.keys()
  checks.sort()

  for check_name in checks:
    doc_file_name = os.path.join(checks_doc_dir, check_name + '.rst')
    #if os.path.exists(doc_file_name):
    #  print('Skipping existing file %s...')
    #  continue
    print('Updating %s...' % doc_file_name)
    with open(doc_file_name, 'w') as f:
      class_name = re.sub(r'.*:', '', registered_checks[check_name])
      f.write(check_name + '\n' + ('=' * len(check_name)) + '\n\n')
      if class_name in defined_checks:
        text = defined_checks[class_name]
        text = re.sub(r'\n//+ ?(\\brief )?', r'\n', text)
        text = re.sub(r'(\n *)\\code\n', r'\1.. code:: c++\n\n', text)
        text = re.sub(r'(\n *)\\endcode(\n|$)', r'\n', text)
        text = re.sub(r'`', r'``', text)
        f.write(text + '\n')
      else:
        f.write('TODO: add docs\n')

  with open(os.path.join(checks_doc_dir, 'list.rst'), 'w') as f:
    f.write(
r"""List of clang-tidy Checks
=========================

.. toctree::
   """ + '\n   '.join(checks))


if __name__ == '__main__':
  main()
