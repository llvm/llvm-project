#!/usr/bin/env python
# A tool to parse the FormatStyle struct from Format.h and update the
# documentation in ../ClangFormatStyleOptions.rst automatically.
# Run from the directory in which this file is located to update the docs.

import collections
import os
import re
import urllib2

CLANG_DIR = os.path.join(os.path.dirname(__file__), '../..')
FORMAT_STYLE_FILE = os.path.join(CLANG_DIR, 'include/clang/Format/Format.h')
DOC_FILE = os.path.join(CLANG_DIR, 'docs/ClangFormatStyleOptions.rst')


def substitute(text, tag, contents):
  replacement = '\n.. START_%s\n\n%s\n\n.. END_%s\n' % (tag, contents, tag)
  pattern = r'\n\.\. START_%s\n.*\n\.\. END_%s\n' % (tag, tag)
  return re.sub(pattern, '%s', text, flags=re.S) % replacement

def doxygen2rst(text):
  text = re.sub(r'<tt>\s*(.*?)\s*<\/tt>', r'``\1``', text)
  text = re.sub(r'\\c ([^ ,;\.]+)', r'``\1``', text)
  text = re.sub(r'\\\w+ ', '', text)
  return text

def indent(text, columns):
  indent = ' ' * columns
  s = re.sub(r'\n([^\n])', '\n' + indent + '\\1', text, flags=re.S)
  if s.startswith('\n'):
    return s
  return indent + s

class Option:
  def __init__(self, name, type, comment):
    self.name = name
    self.type = type
    self.comment = comment.strip()
    self.enum = None
    self.nested_struct = None

  def __str__(self):
    s = '**%s** (``%s``)\n%s' % (self.name, self.type,
                                 doxygen2rst(indent(self.comment, 2)))
    if self.enum:
      s += indent('\n\nPossible values:\n\n%s\n' % self.enum, 2)
    if self.nested_struct:
      s += indent('\n\nNested configuration flags:\n\n%s\n' %self.nested_struct,
                  2)
    return s

class NestedStruct:
  def __init__(self, name, comment):
    self.name = name
    self.comment = comment.strip()
    self.values = []

  def __str__(self):
    return '\n'.join(map(str, self.values))

class NestedField:
  def __init__(self, name, comment):
    self.name = name
    self.comment = comment.strip()

  def __str__(self):
    return '\n* ``%s`` %s' % (self.name, doxygen2rst(self.comment))

class Enum:
  def __init__(self, name, comment):
    self.name = name
    self.comment = comment.strip()
    self.values = []

  def __str__(self):
    return '\n'.join(map(str, self.values))

class EnumValue:
  def __init__(self, name, comment):
    self.name = name
    self.comment = comment

  def __str__(self):
    return '* ``%s`` (in configuration: ``%s``)\n%s' % (
        self.name,
        re.sub('.*_', '', self.name),
        doxygen2rst(indent(self.comment, 2)))

def clean_comment_line(line):
  match = re.match(r'^/// \\code(\{.(\w+)\})?$', line)
  if match:
    lang = match.groups()[1]
    if not lang:
      lang = 'c++'
    return '\n.. code-block:: %s\n\n' % lang
  if line == '/// \\endcode':
    return ''
  return line[4:] + '\n'

def read_options(header):
  class State:
    BeforeStruct, Finished, InStruct, InNestedStruct, InNestedFieldComent, \
    InFieldComment, InEnum, InEnumMemberComment = range(8)
  state = State.BeforeStruct

  options = []
  enums = {}
  nested_structs = {}
  comment = ''
  enum = None
  nested_struct = None

  for line in header:
    line = line.strip()
    if state == State.BeforeStruct:
      if line == 'struct FormatStyle {':
        state = State.InStruct
    elif state == State.InStruct:
      if line.startswith('///'):
        state = State.InFieldComment
        comment = clean_comment_line(line)
      elif line == '};':
        state = State.Finished
        break
    elif state == State.InFieldComment:
      if line.startswith('///'):
        comment += clean_comment_line(line)
      elif line.startswith('enum'):
        state = State.InEnum
        name = re.sub(r'enum\s+(\w+)\s*\{', '\\1', line)
        enum = Enum(name, comment)
      elif line.startswith('struct'):
        state = State.InNestedStruct
        name = re.sub(r'struct\s+(\w+)\s*\{', '\\1', line)
        nested_struct = NestedStruct(name, comment)
      elif line.endswith(';'):
        state = State.InStruct
        field_type, field_name = re.match(r'([<>:\w(,\s)]+)\s+(\w+);',
                                          line).groups()
        option = Option(str(field_name), str(field_type), comment)
        options.append(option)
      else:
        raise Exception('Invalid format, expected comment, field or enum')
    elif state == State.InNestedStruct:
      if line.startswith('///'):
        state = State.InNestedFieldComent
        comment = clean_comment_line(line)
      elif line == '};':
        state = State.InStruct
        nested_structs[nested_struct.name] = nested_struct
    elif state == State.InNestedFieldComent:
      if line.startswith('///'):
        comment += clean_comment_line(line)
      else:
        state = State.InNestedStruct
        nested_struct.values.append(NestedField(line.replace(';', ''), comment))
    elif state == State.InEnum:
      if line.startswith('///'):
        state = State.InEnumMemberComment
        comment = clean_comment_line(line)
      elif line == '};':
        state = State.InStruct
        enums[enum.name] = enum
      else:
        raise Exception('Invalid format, expected enum field comment or };')
    elif state == State.InEnumMemberComment:
      if line.startswith('///'):
        comment += clean_comment_line(line)
      else:
        state = State.InEnum
        enum.values.append(EnumValue(line.replace(',', ''), comment))
  if state != State.Finished:
    raise Exception('Not finished by the end of file')

  for option in options:
    if not option.type in ['bool', 'unsigned', 'int', 'std::string',
                           'std::vector<std::string>',
                           'std::vector<IncludeCategory>']:
      if enums.has_key(option.type):
        option.enum = enums[option.type]
      elif nested_structs.has_key(option.type):
        option.nested_struct = nested_structs[option.type];
      else:
        raise Exception('Unknown type: %s' % option.type)
  return options

options = read_options(open(FORMAT_STYLE_FILE))

options = sorted(options, key=lambda x: x.name)
options_text = '\n\n'.join(map(str, options))

contents = open(DOC_FILE).read()

contents = substitute(contents, 'FORMAT_STYLE_OPTIONS', options_text)

with open(DOC_FILE, 'wb') as output:
  output.write(contents)

