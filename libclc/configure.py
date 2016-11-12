#!/usr/bin/python

def c_compiler_rule(b, name, description, compiler, flags):
  command = "%s -MMD -MF $out.d %s -c -o $out $in" % (compiler, flags)
  b.rule(name, command, description + " $out", depfile="$out.d")

version_major = 0;
version_minor = 2;
version_patch = 0;

from optparse import OptionParser
import os
import string
from subprocess import *
import sys

srcdir = os.path.dirname(sys.argv[0])

sys.path.insert(0, os.path.join(srcdir, 'build'))
import metabuild

p = OptionParser()
p.add_option('--with-llvm-config', metavar='PATH',
             help='use given llvm-config script')
p.add_option('--with-cxx-compiler', metavar='PATH',
             help='use given C++ compiler')
p.add_option('--prefix', metavar='PATH',
             help='install to given prefix')
p.add_option('--libexecdir', metavar='PATH',
             help='install *.bc to given dir')
p.add_option('--includedir', metavar='PATH',
             help='install include files to given dir')
p.add_option('--pkgconfigdir', metavar='PATH',
             help='install clc.pc to given dir')
p.add_option('-g', metavar='GENERATOR', default='make',
             help='use given generator (default: make)')
p.add_option('--enable-runtime-subnormal', action="store_true", default=False,
             help='Allow runtimes to choose subnormal support')
(options, args) = p.parse_args()

llvm_config_exe = options.with_llvm_config or "llvm-config"

prefix = options.prefix
if not prefix:
  prefix = '/usr/local'

libexecdir = options.libexecdir
if not libexecdir:
  libexecdir = os.path.join(prefix, 'lib/clc')

includedir = options.includedir
if not includedir:
  includedir = os.path.join(prefix, 'include')

pkgconfigdir = options.pkgconfigdir
if not pkgconfigdir:
  pkgconfigdir = os.path.join(prefix, 'share/pkgconfig')

def llvm_config(args):
  try:
    proc = Popen([llvm_config_exe] + args, stdout=PIPE)
    return proc.communicate()[0].rstrip().replace('\n', ' ')
  except OSError:
    print "Error executing llvm-config."
    print "Please ensure that llvm-config is in your $PATH, or use --with-llvm-config."
    sys.exit(1)

llvm_version = string.split(string.replace(llvm_config(['--version']), 'svn', ''), '.')
llvm_int_version = int(llvm_version[0]) * 100 + int(llvm_version[1]) * 10
llvm_string_version = 'LLVM' + llvm_version[0] + '.' + llvm_version[1]

if llvm_int_version < 400:
    print "libclc requires LLVM >= 4.0"
    sys.exit(1)

llvm_system_libs = llvm_config(['--system-libs'])
llvm_bindir = llvm_config(['--bindir'])
llvm_core_libs = llvm_config(['--libs', 'core', 'bitreader', 'bitwriter']) + ' ' + \
                 llvm_system_libs + ' ' + \
                 llvm_config(['--ldflags'])
llvm_cxxflags = llvm_config(['--cxxflags']) + ' -fno-exceptions -fno-rtti'
llvm_libdir = llvm_config(['--libdir'])

llvm_clang = os.path.join(llvm_bindir, 'clang')
llvm_link = os.path.join(llvm_bindir, 'llvm-link')
llvm_opt = os.path.join(llvm_bindir, 'opt')

cxx_compiler = options.with_cxx_compiler
if not cxx_compiler:
  cxx_compiler = os.path.join(llvm_bindir, 'clang++')

available_targets = {
  'r600--' : { 'devices' :
               [{'gpu' : 'cedar',   'aliases' : ['palm', 'sumo', 'sumo2', 'redwood', 'juniper']},
                {'gpu' : 'cypress', 'aliases' : ['hemlock'] },
                {'gpu' : 'barts',   'aliases' : ['turks', 'caicos'] },
                {'gpu' : 'cayman',  'aliases' : ['aruba']} ]},
  'amdgcn--': { 'devices' :
                [{'gpu' : 'tahiti', 'aliases' : ['pitcairn', 'verde', 'oland', 'hainan', 'bonaire', 'kabini', 'kaveri', 'hawaii','mullins','tonga','carrizo','iceland','fiji','stoney','polaris10','polaris11']} ]},
  'amdgcn--amdhsa': { 'devices' :
                      [{'gpu' : '', 'aliases' : ['bonaire', 'hawaii', 'kabini', 'kaveri', 'mullins', 'carrizo', 'stoney', 'fiji', 'iceland', 'tonga','polaris10','polaris11']} ]},
  'nvptx--'   : { 'devices' : [{'gpu' : '', 'aliases' : []} ]},
  'nvptx64--' : { 'devices' : [{'gpu' : '', 'aliases' : []} ]},
  'nvptx--nvidiacl'   : { 'devices' : [{'gpu' : '', 'aliases' : []} ]},
  'nvptx64--nvidiacl' : { 'devices' : [{'gpu' : '', 'aliases' : []} ]},
}

available_targets['amdgcn-mesa-mesa3d'] = available_targets['amdgcn--']

default_targets = ['nvptx--nvidiacl', 'nvptx64--nvidiacl', 'r600--', 'amdgcn--', 'amdgcn--amdhsa', 'amdgcn-mesa-mesa3d']

targets = args
if not targets:
  targets = default_targets

b = metabuild.from_name(options.g)

b.rule("LLVM_AS", "%s -o $out $in" % os.path.join(llvm_bindir, "llvm-as"),
       'LLVM-AS $out')
b.rule("LLVM_LINK", command = llvm_link + " -o $out $in",
       description = 'LLVM-LINK $out')
b.rule("OPT", command = llvm_opt + " -O3 -o $out $in",
       description = 'OPT $out')

c_compiler_rule(b, "LLVM_TOOL_CXX", 'CXX', cxx_compiler, llvm_cxxflags)
b.rule("LLVM_TOOL_LINK", cxx_compiler + " -o $out $in %s" % llvm_core_libs + " -Wl,-rpath %s" % llvm_libdir, 'LINK $out')

prepare_builtins = os.path.join('utils', 'prepare-builtins')
b.build(os.path.join('utils', 'prepare-builtins.o'), "LLVM_TOOL_CXX",
        os.path.join(srcdir, 'utils', 'prepare-builtins.cpp'))
b.build(prepare_builtins, "LLVM_TOOL_LINK",
        os.path.join('utils', 'prepare-builtins.o'))

b.rule("PREPARE_BUILTINS", "%s -o $out $in" % prepare_builtins,
       'PREPARE-BUILTINS $out')
b.rule("PYTHON_GEN", "python < $in > $out", "PYTHON_GEN $out")
b.build('generic/lib/convert.cl', "PYTHON_GEN", ['generic/lib/gen_convert.py'])

manifest_deps = set([sys.argv[0], os.path.join(srcdir, 'build', 'metabuild.py'),
                     os.path.join(srcdir, 'build', 'ninja_syntax.py')])

install_files_bc = []
install_deps = []

# Create rules for subnormal helper objects
for src in ['subnormal_disable.ll', 'subnormal_use_default.ll']:
  obj_name = src[:-2] + 'bc'
  obj = os.path.join('generic--', 'lib', obj_name)
  src_file = os.path.join('generic', 'lib', src)
  b.build(obj, 'LLVM_AS', src_file)
  b.default(obj)
  install_files_bc.append((obj, obj))
  install_deps.append(obj)

# Create libclc.pc
clc = open('libclc.pc', 'w')
clc.write('includedir=%(inc)s\nlibexecdir=%(lib)s\n\nName: libclc\nDescription: Library requirements of the OpenCL C programming language\nVersion: %(maj)s.%(min)s.%(pat)s\nCflags: -I${includedir}\nLibs: -L${libexecdir}' %
{'inc': includedir, 'lib': libexecdir, 'maj': version_major, 'min': version_minor, 'pat': version_patch})
clc.close()

for target in targets:
  (t_arch, t_vendor, t_os) = target.split('-')
  archs = [t_arch]
  if t_arch == 'nvptx' or t_arch == 'nvptx64':
    archs.append('ptx')
  archs.append('generic')

  subdirs = []
  for arch in archs:
    subdirs.append("%s-%s-%s" % (arch, t_vendor, t_os))
    subdirs.append("%s-%s" % (arch, t_os))
    if t_os == 'mesa3d':
        subdirs.append('amdgcn-amdhsa')
    subdirs.append(arch)
    if arch == 'amdgcn' or arch == 'r600':
        subdirs.append('amdgpu')

  incdirs = filter(os.path.isdir,
               [os.path.join(srcdir, subdir, 'include') for subdir in subdirs])
  libdirs = filter(lambda d: os.path.isfile(os.path.join(d, 'SOURCES')),
                   [os.path.join(srcdir, subdir, 'lib') for subdir in subdirs])

  clang_cl_includes = ' '.join(["-I%s" % incdir for incdir in incdirs])

  for device in available_targets[target]['devices']:
    # The rule for building a .bc file for the specified architecture using clang.
    clang_bc_flags = "-target %s -I`dirname $in` %s " \
                     "-fno-builtin " \
                     "-D__CLC_INTERNAL " \
                     "-emit-llvm" % (target, clang_cl_includes)
    if device['gpu'] != '':
      clang_bc_flags += ' -mcpu=' + device['gpu']
    clang_bc_rule = "CLANG_CL_BC_" + target + "_" + device['gpu']
    c_compiler_rule(b, clang_bc_rule, "LLVM-CC", llvm_clang, clang_bc_flags)

    objects = []
    sources_seen = set()
    compats_seen = set()

    if device['gpu'] == '':
      full_target_name = target
      obj_suffix = ''
    else:
      full_target_name = device['gpu'] + '-' + target
      obj_suffix = '.' + device['gpu']

    for libdir in libdirs:
      subdir_list_file = os.path.join(libdir, 'SOURCES')
      manifest_deps.add(subdir_list_file)
      override_list_file = os.path.join(libdir, 'OVERRIDES')
      compat_list_file = os.path.join(libdir,
        'SOURCES_' + llvm_string_version)

      # Build compat list
      if os.path.exists(compat_list_file):
        for compat in open(compat_list_file).readlines():
          compat = compat.rstrip()
          compats_seen.add(compat)

      # Add target overrides
      if os.path.exists(override_list_file):
        for override in open(override_list_file).readlines():
          override = override.rstrip()
          sources_seen.add(override)

      for src in open(subdir_list_file).readlines():
        src = src.rstrip()
        if src not in sources_seen:
          sources_seen.add(src)
          obj = os.path.join(target, 'lib', src + obj_suffix + '.bc')
          objects.append(obj)
          src_path = libdir
          if src in compats_seen:
            src_path = os.path.join(libdir, llvm_string_version)
          src_file = os.path.join(src_path, src)
          ext = os.path.splitext(src)[1]
          if ext == '.ll':
            b.build(obj, 'LLVM_AS', src_file)
          else:
            b.build(obj, clang_bc_rule, src_file)

    obj = os.path.join('generic--', 'lib', 'subnormal_use_default.bc')
    if  not options.enable_runtime_subnormal:
      objects.append(obj)

    builtins_link_bc = os.path.join(target, 'lib', 'builtins.link' + obj_suffix + '.bc')
    builtins_opt_bc = os.path.join(target, 'lib', 'builtins.opt' + obj_suffix + '.bc')
    builtins_bc = os.path.join('built_libs', full_target_name + '.bc')
    b.build(builtins_link_bc, "LLVM_LINK", objects)
    b.build(builtins_opt_bc, "OPT", builtins_link_bc)
    b.build(builtins_bc, "PREPARE_BUILTINS", builtins_opt_bc, prepare_builtins)
    install_files_bc.append((builtins_bc, builtins_bc))
    install_deps.append(builtins_bc)
    for alias in device['aliases']:
      # Ninja cannot have multiple rules with same name so append suffix
      ruleName = "CREATE_ALIAS_{0}_for_{1}".format(alias, device['gpu'])
      b.rule(ruleName, "ln -fs %s $out" % os.path.basename(builtins_bc)
             ,"CREATE-ALIAS $out")

      alias_file = os.path.join('built_libs', alias + '-' + target + '.bc')
      b.build(alias_file, ruleName, builtins_bc)
      install_files_bc.append((alias_file, alias_file))
      install_deps.append(alias_file)
    b.default(builtins_bc)


install_cmd = ' && '.join(['mkdir -p ${DESTDIR}/%(dst)s && cp -r %(src)s ${DESTDIR}/%(dst)s' %
                           {'src': file,
                            'dst': libexecdir}
                           for (file, dest) in install_files_bc])
install_cmd = ' && '.join(['%(old)s && mkdir -p ${DESTDIR}/%(dst)s && cp -r %(srcdir)s/generic/include/clc ${DESTDIR}/%(dst)s' %
                           {'old': install_cmd,
                            'dst': includedir,
                            'srcdir': srcdir}])
install_cmd = ' && '.join(['%(old)s && mkdir -p ${DESTDIR}/%(dst)s && cp -r libclc.pc ${DESTDIR}/%(dst)s' %
                           {'old': install_cmd, 
                            'dst': pkgconfigdir}])
  
b.rule('install', command = install_cmd, description = 'INSTALL')
b.build('install', 'install', install_deps)

b.rule("configure", command = ' '.join(sys.argv), description = 'CONFIGURE',
       generator = True)
b.build(b.output_filename(), 'configure', list(manifest_deps))

b.finish()
