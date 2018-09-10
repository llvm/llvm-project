import ninja_syntax
import os

# Simple meta-build system.

class Make(object):
  def __init__(self):
    self.output = open(self.output_filename(), 'w')
    self.rules = {}
    self.rule_text = ''
    self.all_targets = []
    self.default_targets = []
    self.clean_files = []
    self.distclean_files = []
    self.output.write("""all::

ifndef VERBOSE
  Verb = @
endif

""")

  def output_filename(self):
    return 'Makefile'

  def rule(self, name, command, description=None, depfile=None,
           generator=False):
    self.rules[name] = {'command': command, 'description': description,
                        'depfile': depfile, 'generator': generator}

  def build(self, output, rule, inputs=[], implicit=[], order_only=[]):
    inputs = self._as_list(inputs)
    implicit = self._as_list(implicit)
    order_only = self._as_list(order_only)

    output_dir = os.path.dirname(output)
    if output_dir != '' and not os.path.isdir(output_dir):
      os.makedirs(output_dir)

    dollar_in = ' '.join(inputs)
    subst = lambda text: text.replace('$in', dollar_in).replace('$out', output)

    deps = ' '.join(inputs + implicit)
    if order_only:
      deps += ' | '
      deps += ' '.join(order_only)
    self.output.write('%s: %s\n' % (output, deps))

    r = self.rules[rule]
    command = subst(r['command'])
    if r['description']:
      desc = subst(r['description'])
      self.output.write('\t@echo %s\n\t$(Verb) %s\n' % (desc, command))
    else:
      self.output.write('\t%s\n' % command)
    if r['depfile']:
      depfile = subst(r['depfile'])
      self.output.write('-include '+depfile+'\n')
    self.output.write('\n')

    self.all_targets.append(output)
    if r['generator']:
      self.distclean_files.append(output)
      if r['depfile']:
        self.distclean_files.append(depfile)
    else:
      self.clean_files.append(output)
      if r['depfile']:
        self.distclean_files.append(depfile)


  def _as_list(self, input):
    if isinstance(input, list):
      return input
    return [input]

  def default(self, paths):
    self.default_targets += self._as_list(paths)

  def finish(self):
    self.output.write('all:: %s\n\n' % ' '.join(self.default_targets or self.all_targets))
    self.output.write('clean: \n\trm -f %s\n\n' % ' '.join(self.clean_files))
    self.output.write('distclean: clean\n\trm -f %s\n' % ' '.join(self.distclean_files))

class Ninja(ninja_syntax.Writer):
  def __init__(self):
    ninja_syntax.Writer.__init__(self, open(self.output_filename(), 'w'))

  def output_filename(self):
    return 'build.ninja'

  def finish(self):
    pass

def from_name(name):
  if name == 'make':
    return Make()
  if name == 'ninja':
    return Ninja()
  raise LookupError('unknown generator: %s; supported generators are make and ninja' % name)
