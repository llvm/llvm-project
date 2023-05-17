#!/usr/bin/env python3
#
#===- add_new_check.py - clang-tidy check generator ---------*- python -*--===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===-----------------------------------------------------------------------===#

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import io
import os
import re
import sys

# Adapts the module's CMakelist file. Returns 'True' if it could add a new
# entry and 'False' if the entry already existed.
def adapt_cmake(module_path, check_name_camel):
  filename = os.path.join(module_path, 'CMakeLists.txt')

  # The documentation files are encoded using UTF-8, however on Windows the
  # default encoding might be different (e.g. CP-1252). To make sure UTF-8 is
  # always used, use `io.open(filename, mode, encoding='utf8')` for reading and
  # writing files here and elsewhere.
  with io.open(filename, 'r', encoding='utf8') as f:
    lines = f.readlines()

  cpp_file = check_name_camel + '.cpp'

  # Figure out whether this check already exists.
  for line in lines:
    if line.strip() == cpp_file:
      return False

  print('Updating %s...' % filename)
  with io.open(filename, 'w', encoding='utf8', newline='\n') as f:
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


# Adds a header for the new check.
def write_header(module_path, module, namespace, check_name, check_name_camel):
  filename = os.path.join(module_path, check_name_camel) + '.h'
  print('Creating %s...' % filename)
  with io.open(filename, 'w', encoding='utf8', newline='\n') as f:
    header_guard = ('LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_' + module.upper() + '_'
                    + check_name_camel.upper() + '_H')
    f.write('//===--- ')
    f.write(os.path.basename(filename))
    f.write(' - clang-tidy ')
    f.write('-' * max(0, 42 - len(os.path.basename(filename))))
    f.write('*- C++ -*-===//')
    f.write("""
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef %(header_guard)s
#define %(header_guard)s

#include "../ClangTidyCheck.h"

namespace clang::tidy::%(namespace)s {

/// FIXME: Write a short description.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/%(module)s/%(check_name)s.html
class %(check_name_camel)s : public ClangTidyCheck {
public:
  %(check_name_camel)s(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::%(namespace)s

#endif // %(header_guard)s
""" % {'header_guard': header_guard,
       'check_name_camel': check_name_camel,
       'check_name': check_name,
       'module': module,
       'namespace': namespace})


# Adds the implementation of the new check.
def write_implementation(module_path, module, namespace, check_name_camel):
  filename = os.path.join(module_path, check_name_camel) + '.cpp'
  print('Creating %s...' % filename)
  with io.open(filename, 'w', encoding='utf8', newline='\n') as f:
    f.write('//===--- ')
    f.write(os.path.basename(filename))
    f.write(' - clang-tidy ')
    f.write('-' * max(0, 51 - len(os.path.basename(filename))))
    f.write('-===//')
    f.write("""
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "%(check_name)s.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::%(namespace)s {

void %(check_name)s::registerMatchers(MatchFinder *Finder) {
  // FIXME: Add matchers.
  Finder->addMatcher(functionDecl().bind("x"), this);
}

void %(check_name)s::check(const MatchFinder::MatchResult &Result) {
  // FIXME: Add callback implementation.
  const auto *MatchedDecl = Result.Nodes.getNodeAs<FunctionDecl>("x");
  if (!MatchedDecl->getIdentifier() || MatchedDecl->getName().startswith("awesome_"))
    return;
  diag(MatchedDecl->getLocation(), "function %%0 is insufficiently awesome")
      << MatchedDecl
      << FixItHint::CreateInsertion(MatchedDecl->getLocation(), "awesome_");
  diag(MatchedDecl->getLocation(), "insert 'awesome'", DiagnosticIDs::Note);
}

} // namespace clang::tidy::%(namespace)s
""" % {'check_name': check_name_camel,
       'module': module,
       'namespace': namespace})


# Returns the source filename that implements the module.
def get_module_filename(module_path, module):
  modulecpp = list(filter(
      lambda p: p.lower() == module.lower() + 'tidymodule.cpp',
      os.listdir(module_path)))[0]
  return os.path.join(module_path, modulecpp)


# Modifies the module to include the new check.
def adapt_module(module_path, module, check_name, check_name_camel):
  filename = get_module_filename(module_path, module)
  with io.open(filename, 'r', encoding='utf8') as f:
    lines = f.readlines()

  print('Updating %s...' % filename)
  with io.open(filename, 'w', encoding='utf8', newline='\n') as f:
    header_added = False
    header_found = False
    check_added = False
    check_fq_name = module + '-' + check_name
    check_decl = ('    CheckFactories.registerCheck<' + check_name_camel +
                  '>(\n        "' + check_fq_name + '");\n')

    lines = iter(lines)
    try:
      while True:
        line = next(lines)
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
            match = re.search('registerCheck<(.*)> *\( *(?:"([^"]*)")?', line)
            prev_line = None
            if match:
              current_check_name = match.group(2)
              if current_check_name is None:
                # If we didn't find the check name on this line, look on the
                # next one.
                prev_line = line
                line = next(lines)
                match = re.search(' *"([^"]*)"', line)
                if match:
                  current_check_name = match.group(1)
              if current_check_name > check_fq_name:
                check_added = True
                f.write(check_decl)
              if prev_line:
                f.write(prev_line)
        f.write(line)
    except StopIteration:
      pass


# Adds a release notes entry.
def add_release_notes(module_path, module, check_name):
  check_name_dashes = module + '-' + check_name
  filename = os.path.normpath(os.path.join(module_path,
                                           '../../docs/ReleaseNotes.rst'))
  with io.open(filename, 'r', encoding='utf8') as f:
    lines = f.readlines()

  lineMatcher = re.compile('New checks')
  nextSectionMatcher = re.compile('New check aliases')
  checkMatcher = re.compile('- New :doc:`(.*)')

  print('Updating %s...' % filename)
  with io.open(filename, 'w', encoding='utf8', newline='\n') as f:
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
          if last_check > check_name_dashes:
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
            f.write("""- New :doc:`%s
  <clang-tidy/checks/%s/%s>` check.

  FIXME: add release notes.

""" % (check_name_dashes, module, check_name))
            note_added = True

      f.write(line)


# Adds a test for the check.
def write_test(module_path, module, check_name, test_extension):
  check_name_dashes = module + '-' + check_name
  filename = os.path.normpath(os.path.join(
    module_path, '..', '..', 'test', 'clang-tidy', 'checkers',
    module, check_name + '.' + test_extension))
  print('Creating %s...' % filename)
  with io.open(filename, 'w', encoding='utf8', newline='\n') as f:
    f.write("""// RUN: %%check_clang_tidy %%s %(check_name_dashes)s %%t

// FIXME: Add something that triggers the check here.
void f();
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'f' is insufficiently awesome [%(check_name_dashes)s]

// FIXME: Verify the applied fix.
//   * Make the CHECK patterns specific enough and try to make verified lines
//     unique to avoid incorrect matches.
//   * Use {{}} for regular expressions.
// CHECK-FIXES: {{^}}void awesome_f();{{$}}

// FIXME: Add something that doesn't trigger the check here.
void awesome_f2();
""" % {'check_name_dashes': check_name_dashes})


def get_actual_filename(dirname, filename):
  if not os.path.isdir(dirname):
    return ''
  name = os.path.join(dirname, filename)
  if (os.path.isfile(name)):
    return name
  caselessname = filename.lower()
  for file in os.listdir(dirname):
    if (file.lower() == caselessname):
      return os.path.join(dirname, file)
  return ''


# Recreates the list of checks in the docs/clang-tidy/checks directory.
def update_checks_list(clang_tidy_path):
  docs_dir = os.path.join(clang_tidy_path, '../docs/clang-tidy/checks')
  filename = os.path.normpath(os.path.join(docs_dir, 'list.rst'))
  # Read the content of the current list.rst file
  with io.open(filename, 'r', encoding='utf8') as f:
    lines = f.readlines()
  # Get all existing docs
  doc_files = []
  for subdir in filter(lambda s: os.path.isdir(os.path.join(docs_dir, s)), os.listdir(docs_dir)):
    for file in filter(lambda s: s.endswith('.rst'), os.listdir(os.path.join(docs_dir, subdir))):
      doc_files.append([subdir, file])
  doc_files.sort()

  # We couldn't find the source file from the check name, so try to find the
  # class name that corresponds to the check in the module file.
  def filename_from_module(module_name, check_name):
    module_path = os.path.join(clang_tidy_path, module_name)
    if not os.path.isdir(module_path):
      return ''
    module_file = get_module_filename(module_path, module_name)
    if not os.path.isfile(module_file):
      return ''
    with io.open(module_file, 'r') as f:
      code = f.read()
      full_check_name = module_name + '-' + check_name
      name_pos = code.find('"' + full_check_name + '"')
      if name_pos == -1:
        return ''
      stmt_end_pos = code.find(';', name_pos)
      if stmt_end_pos == -1:
        return ''
      stmt_start_pos = code.rfind(';', 0, name_pos)
      if stmt_start_pos == -1:
        stmt_start_pos = code.rfind('{', 0, name_pos)
      if stmt_start_pos == -1:
        return ''
      stmt = code[stmt_start_pos+1:stmt_end_pos]
      matches = re.search('registerCheck<([^>:]*)>\(\s*"([^"]*)"\s*\)', stmt)
      if matches and matches[2] == full_check_name:
        class_name = matches[1]
        if '::' in class_name:
          parts = class_name.split('::')
          class_name = parts[-1]
          class_path = os.path.join(clang_tidy_path, module_name, '..', *parts[0:-1])
        else:
          class_path = os.path.join(clang_tidy_path, module_name)
        return get_actual_filename(class_path, class_name + '.cpp')

    return ''

  # Examine code looking for a c'tor definition to get the base class name.
  def get_base_class(code, check_file):
    check_class_name = os.path.splitext(os.path.basename(check_file))[0]
    ctor_pattern = check_class_name + '\([^:]*\)\s*:\s*([A-Z][A-Za-z0-9]*Check)\('
    matches = re.search('\s+' + check_class_name + '::' + ctor_pattern, code)

    # The constructor might be inline in the header.
    if not matches:
      header_file = os.path.splitext(check_file)[0] + '.h'
      if not os.path.isfile(header_file):
        return ''
      with io.open(header_file, encoding='utf8') as f:
        code = f.read()
      matches = re.search(' ' + ctor_pattern, code)

    if matches and matches[1] != 'ClangTidyCheck':
      return matches[1]
    return ''

  # Some simple heuristics to figure out if a check has an autofix or not.
  def has_fixits(code):
    for needle in ['FixItHint', 'ReplacementText', 'fixit',
                   'TransformerClangTidyCheck']:
      if needle in code:
        return True
    return False

  # Try to figure out of the check supports fixits.
  def has_auto_fix(check_name):
    dirname, _, check_name = check_name.partition('-')

    check_file = get_actual_filename(os.path.join(clang_tidy_path, dirname),
                                       get_camel_check_name(check_name) + '.cpp')
    if not os.path.isfile(check_file):
      # Some older checks don't end with 'Check.cpp'
      check_file = get_actual_filename(os.path.join(clang_tidy_path, dirname),
                                         get_camel_name(check_name) + '.cpp')
      if not os.path.isfile(check_file):
        # Some checks aren't in a file based on the check name.
        check_file = filename_from_module(dirname, check_name)
        if not check_file or not os.path.isfile(check_file):
          return ''

    with io.open(check_file, encoding='utf8') as f:
      code = f.read()
      if has_fixits(code):
        return ' "Yes"'

    base_class = get_base_class(code, check_file)
    if base_class:
      base_file = os.path.join(clang_tidy_path, dirname, base_class + '.cpp')
      if os.path.isfile(base_file):
        with io.open(base_file, encoding='utf8') as f:
          code = f.read()
          if has_fixits(code):
            return ' "Yes"'

    return ''

  def process_doc(doc_file):
    check_name = doc_file[0] + '-' + doc_file[1].replace('.rst', '')

    with io.open(os.path.join(docs_dir, *doc_file), 'r', encoding='utf8') as doc:
      content = doc.read()
      match = re.search('.*:orphan:.*', content)

      if match:
        # Orphan page, don't list it.
        return '', ''

      match = re.search('.*:http-equiv=refresh: \d+;URL=(.*).html(.*)',
                        content)
      # Is it a redirect?
      return check_name, match

  def format_link(doc_file):
    check_name, match = process_doc(doc_file)
    if not match and check_name:
      return '   `%(check_name)s <%(module)s/%(check)s.html>`_,%(autofix)s\n' % {
        'check_name': check_name,
        'module': doc_file[0],
        'check': doc_file[1].replace('.rst', ''),
        'autofix': has_auto_fix(check_name)
      }
    else:
      return ''

  def format_link_alias(doc_file):
    check_name, match = process_doc(doc_file)
    if match and check_name:
      module = doc_file[0]
      check_file = doc_file[1].replace('.rst', '')
      if match.group(1) == 'https://clang.llvm.org/docs/analyzer/checkers':
        title = 'Clang Static Analyzer ' + check_file
        # Preserve the anchor in checkers.html from group 2.
        target = match.group(1) + '.html' + match.group(2)
        autofix = ''
      else:
        redirect_parts = re.search('^\.\./([^/]*)/([^/]*)$', match.group(1))
        title = redirect_parts[1] + '-' + redirect_parts[2]
        target = redirect_parts[1] + '/' + redirect_parts[2] + '.html'
        autofix = has_auto_fix(title)

      # The checker is just a redirect.
      return '   `%(check_name)s <%(module)s/%(check_file)s.html>`_, `%(title)s <%(target)s>`_,%(autofix)s\n' % {
        'check_name': check_name,
        'module': module,
        'check_file': check_file,
        'target': target,
        'title': title,
        'autofix': autofix
      }
    return ''

  checks = map(format_link, doc_files)
  checks_alias = map(format_link_alias, doc_files)

  print('Updating %s...' % filename)
  with io.open(filename, 'w', encoding='utf8', newline='\n') as f:
    for line in lines:
      f.write(line)
      if line.strip() == '.. csv-table::':
        # We dump the checkers
        f.write('   :header: "Name", "Offers fixes"\n\n')
        f.writelines(checks)
        # and the aliases
        f.write('\n\n')
        f.write('.. csv-table:: Aliases..\n')
        f.write('   :header: "Name", "Redirect", "Offers fixes"\n\n')
        f.writelines(checks_alias)
        break


# Adds a documentation for the check.
def write_docs(module_path, module, check_name):
  check_name_dashes = module + '-' + check_name
  filename = os.path.normpath(os.path.join(
      module_path, '../../docs/clang-tidy/checks/', module, check_name + '.rst'))
  print('Creating %s...' % filename)
  with io.open(filename, 'w', encoding='utf8', newline='\n') as f:
    f.write(""".. title:: clang-tidy - %(check_name_dashes)s

%(check_name_dashes)s
%(underline)s

FIXME: Describe what patterns does the check detect and why. Give examples.
""" % {'check_name_dashes': check_name_dashes,
       'underline': '=' * len(check_name_dashes)})


def get_camel_name(check_name):
  return ''.join(map(lambda elem: elem.capitalize(),
                     check_name.split('-')))


def get_camel_check_name(check_name):
  return get_camel_name(check_name) + 'Check'


def main():
  language_to_extension = {
      'c': 'c',
      'c++': 'cpp',
      'objc': 'm',
      'objc++': 'mm',
  }
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--update-docs',
      action='store_true',
      help='just update the list of documentation files, then exit')
  parser.add_argument(
      '--language',
      help='language to use for new check (defaults to c++)',
      choices=language_to_extension.keys(),
      default='c++',
      metavar='LANG')
  parser.add_argument(
      'module',
      nargs='?',
      help='module directory under which to place the new tidy check (e.g., misc)')
  parser.add_argument(
      'check',
      nargs='?',
      help='name of new tidy check to add (e.g. foo-do-the-stuff)')
  args = parser.parse_args()

  if args.update_docs:
    update_checks_list(os.path.dirname(sys.argv[0]))
    return

  if not args.module or not args.check:
    print('Module and check must be specified.')
    parser.print_usage()
    return

  module = args.module
  check_name = args.check
  check_name_camel = get_camel_check_name(check_name)
  if check_name.startswith(module):
    print('Check name "%s" must not start with the module "%s". Exiting.' % (
        check_name, module))
    return
  clang_tidy_path = os.path.dirname(sys.argv[0])
  module_path = os.path.join(clang_tidy_path, module)

  if not adapt_cmake(module_path, check_name_camel):
    return

  # Map module names to namespace names that don't conflict with widely used top-level namespaces.
  if module == 'llvm':
    namespace = module + '_check'
  else:
    namespace = module

  write_header(module_path, module, namespace, check_name, check_name_camel)
  write_implementation(module_path, module, namespace, check_name_camel)
  adapt_module(module_path, module, check_name, check_name_camel)
  add_release_notes(module_path, module, check_name)
  test_extension = language_to_extension.get(args.language)
  write_test(module_path, module, check_name, test_extension)
  write_docs(module_path, module, check_name)
  update_checks_list(clang_tidy_path)
  print('Done. Now it\'s your turn!')


if __name__ == '__main__':
  main()
