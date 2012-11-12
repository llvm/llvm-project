#!/usr/bin/env python
# A tool to parse ASTMatchers.h and update the documentation in
# ../LibASTMatchersReference.html automatically. Run from the
# directory in which this file is located to update the docs.

import collections
import re
import urllib2

MATCHERS_FILE = '../../include/clang/ASTMatchers/ASTMatchers.h'

# Each matcher is documented in one row of the form:
#   result | name | argA
# The subsequent row contains the documentation and is hidden by default,
# becoming visible via javascript when the user clicks the matcher name.
TD_TEMPLATE="""
<tr><td>%(result)s</td><td class="name" onclick="toggle('%(id)s')"><a name="%(id)sAnchor">%(name)s</a></td><td>%(args)s</td></tr>
<tr><td colspan="4" class="doc" id="%(id)s"><pre>%(comment)s</pre></td></tr>
"""

# We categorize the matchers into these three categories in the reference:
node_matchers = {}
narrowing_matchers = {}
traversal_matchers = {}

# We output multiple rows per matcher if the matcher can be used on multiple
# node types. Thus, we need a new id per row to control the documentation
# pop-up. ids[name] keeps track of those ids.
ids = collections.defaultdict(int)

# Cache for doxygen urls we have already verified.
doxygen_probes = {}

def esc(text):
  """Escape any html in the given text."""
  text = re.sub(r'&', '&amp;', text)
  text = re.sub(r'<', '&lt;', text)
  text = re.sub(r'>', '&gt;', text)
  def link_if_exists(m):
    name = m.group(1)
    url = 'http://clang.llvm.org/doxygen/classclang_1_1%s.html' % name
    if url not in doxygen_probes:
      try:
        print 'Probing %s...' % url
        urllib2.urlopen(url)
        doxygen_probes[url] = True
      except:
        doxygen_probes[url] = False
    if doxygen_probes[url]:
      return r'Matcher&lt<a href="%s">%s</a>&gt;' % (url, name)
    else:
      return m.group(0)
  text = re.sub(
    r'Matcher&lt;([^\*&]+)&gt;', link_if_exists, text)
  return text

def extract_result_types(comment):
  """Extracts a list of result types from the given comment.

     We allow annotations in the comment of the matcher to specify what
     nodes a matcher can match on. Those comments have the form:
       Usable as: Any Matcher | (Matcher<T1>[, Matcher<t2>[, ...]])

     Returns ['*'] in case of 'Any Matcher', or ['T1', 'T2', ...].
     Returns the empty list if no 'Usable as' specification could be
     parsed.
  """
  result_types = []
  m = re.search(r'Usable as: Any Matcher[\s\n]*$', comment, re.S)
  if m:
    return ['*']
  while True:
    m = re.match(r'^(.*)Matcher<([^>]+)>\s*,?[\s\n]*$', comment, re.S)
    if not m:
      if re.search(r'Usable as:\s*$', comment):
        return result_types
      else:
        return None
    result_types += [m.group(2)]
    comment = m.group(1)

def strip_doxygen(comment):
  """Returns the given comment without \-escaped words."""
  # If there is only a doxygen keyword in the line, delete the whole line.
  comment = re.sub(r'^\\[^\s]+\n', r'', comment, flags=re.M)
  # Delete the doxygen command and the following whitespace.
  comment = re.sub(r'\\[^\s]+\s+', r'', comment)
  return comment

def unify_arguments(args):
  """Gets rid of anything the user doesn't care about in the argument list."""
  args = re.sub(r'internal::', r'', args)
  args = re.sub(r'const\s+', r'', args)
  args = re.sub(r'&', r' ', args)
  args = re.sub(r'(^|\s)M\d?(\s)', r'\1Matcher<*>\2', args)
  return args

def add_matcher(result_type, name, args, comment, is_dyncast=False):
  """Adds a matcher to one of our categories."""
  if name == 'id':
     # FIXME: Figure out whether we want to support the 'id' matcher.
     return
  matcher_id = '%s%d' % (name, ids[name])
  ids[name] += 1
  args = unify_arguments(args)
  matcher_html = TD_TEMPLATE % {
    'result': esc('Matcher<%s>' % result_type),
    'name': name,
    'args': esc(args),
    'comment': esc(strip_doxygen(comment)),
    'id': matcher_id,
  }
  if is_dyncast:
    node_matchers[result_type + name] = matcher_html
  # Use a heuristic to figure out whether a matcher is a narrowing or
  # traversal matcher. By default, matchers that take other matchers as
  # arguments (and are not node matchers) do traversal. We specifically
  # exclude known narrowing matchers that also take other matchers as
  # arguments.
  elif ('Matcher<' not in args or
        name in ['allOf', 'anyOf', 'anything', 'unless']):
    narrowing_matchers[result_type + name] = matcher_html
  else:
    traversal_matchers[result_type + name] = matcher_html

def act_on_decl(declaration, comment, allowed_types):
  """Parse the matcher out of the given declaration and comment.

     If 'allowed_types' is set, it contains a list of node types the matcher
     can match on, as extracted from the static type asserts in the matcher
     definition.
  """
  if declaration.strip():
    # Node matchers are defined by writing:
    #   VariadicDynCastAllOfMatcher<ResultType, ArgumentType> name;
    m = re.match(r""".*VariadicDynCastAllOfMatcher\s*<
                       \s*([^\s,]+)\s*,
                       \s*([^\s>]+)\s*>
                       \s*([^\s;]+)\s*;\s*$""", declaration, flags=re.X)
    if m:
      result, inner, name = m.groups()
      add_matcher(result, name, 'Matcher<%s>...' % inner,
                  comment, is_dyncast=True)
      return

    # Parse the various matcher definition macros.
    m = re.match(r"""^\s*AST_(POLYMORPHIC_)?MATCHER(_P)?(.?)\(
                       (?:\s*([^\s,]+)\s*,)?
                          \s*([^\s,]+)\s*
                       (?:,\s*([^\s,]+)\s*
                          ,\s*([^\s,]+)\s*)?
                       (?:,\s*([^\s,]+)\s*
                          ,\s*([^\s,]+)\s*)?
                      \)\s*{\s*$""", declaration, flags=re.X)
    if m:
      p, n, result, name = m.groups()[1:5]
      args = m.groups()[5:]
      if not result:
        if not allowed_types:
          raise Exception('Did not find allowed result types for: %s' % name)
        result_types = allowed_types
      else:
        result_types = [result]
      if n not in ['', '2']:
        raise Exception('Cannot parse "%s"' % declaration)
      args = ', '.join('%s %s' % (args[i], args[i+1])
                       for i in range(0, len(args), 2) if args[i])
      for result_type in result_types:
        add_matcher(result_type, name, args, comment)
      return

    # Parse free standing matcher functions, like:
    #   Matcher<ResultType> Name(Matcher<ArgumentType> InnerMatcher) {
    m = re.match(r"""^\s*(.*)\s+
                     ([^\s\(]+)\s*\(
                     (.*)
                     \)\s*{""", declaration, re.X)
    if m:
      result, name, args = m.groups()
      args = ', '.join(p.strip() for p in args.split(','))
      m = re.match(r'.*\s+internal::Matcher<([^>]+)>$', result)
      if m:
        result_types = [m.group(1)]
      else:
        result_types = extract_result_types(comment)
      if not result_types:
        if not comment:
          # Only overloads don't have their own doxygen comments; ignore those.
          print 'Ignoring "%s"' % name
        else:
          print 'Cannot determine result type for "%s"' % name
      else:
        for result_type in result_types:
          add_matcher(result_type, name, args, comment)
    else:
      print '*** Unparsable: "' + declaration + '" ***'

def sort_table(matcher_type, matcher_map):
  """Returns the sorted html table for the given row map."""
  table = ''
  for key in sorted(matcher_map.keys()):
    table += matcher_map[key] + '\n'
  return ('<!-- START_%(type)s_MATCHERS -->\n' +
          '%(table)s' + 
          '<!--END_%(type)s_MATCHERS -->') % {
    'type': matcher_type,
    'table': table,
  }

# Parse the ast matchers.
# We alternate between two modes:
# body = True: We parse the definition of a matcher. We need
#   to parse the full definition before adding a matcher, as the
#   definition might contain static asserts that specify the result
#   type.
# body = False: We parse the comments and declaration of the matcher.
comment = ''
declaration = ''
allowed_types = []
body = False
for line in open(MATCHERS_FILE).read().splitlines():
  if body:
    if line.strip() and line[0] == '}':
      if declaration:
        act_on_decl(declaration, comment, allowed_types)
        comment = ''
        declaration = ''
        allowed_types = []
      body = False
    else:
      m = re.search(r'is_base_of<([^,]+), NodeType>', line)
      if m and m.group(1):
        allowed_types += [m.group(1)]
    continue
  if line.strip() and line.lstrip()[0] == '/':
    comment += re.sub(r'/+\s?', '', line) + '\n'
  else:
    declaration += ' ' + line
    if ((not line.strip()) or 
        line.rstrip()[-1] == ';' or
        line.rstrip()[-1] == '{'):
      if line.strip() and line.rstrip()[-1] == '{':
        body = True
      else:
        act_on_decl(declaration, comment, allowed_types)
        comment = ''
        declaration = ''
        allowed_types = []

node_matcher_table = sort_table('DECL', node_matchers)
narrowing_matcher_table = sort_table('NARROWING', narrowing_matchers)
traversal_matcher_table = sort_table('TRAVERSAL', traversal_matchers)

reference = open('../LibASTMatchersReference.html').read()
reference = re.sub(r'<!-- START_DECL_MATCHERS.*END_DECL_MATCHERS -->',
                   '%s', reference, flags=re.S) % node_matcher_table
reference = re.sub(r'<!-- START_NARROWING_MATCHERS.*END_NARROWING_MATCHERS -->',
                   '%s', reference, flags=re.S) % narrowing_matcher_table
reference = re.sub(r'<!-- START_TRAVERSAL_MATCHERS.*END_TRAVERSAL_MATCHERS -->',
                   '%s', reference, flags=re.S) % traversal_matcher_table

with open('../LibASTMatchersReference.html', 'w') as output:
  output.write(reference)

