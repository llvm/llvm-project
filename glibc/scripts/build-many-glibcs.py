#!/usr/bin/python3
# Build many configurations of glibc.
# Copyright (C) 2016-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
#
# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

"""Build many configurations of glibc.

This script takes as arguments a directory name (containing a src
subdirectory with sources of the relevant toolchain components) and a
description of what to do: 'checkout', to check out sources into that
directory, 'bot-cycle', to run a series of checkout and build steps,
'bot', to run 'bot-cycle' repeatedly, 'host-libraries', to build
libraries required by the toolchain, 'compilers', to build
cross-compilers for various configurations, or 'glibcs', to build
glibc for various configurations and run the compilation parts of the
testsuite.  Subsequent arguments name the versions of components to
check out (<component>-<version), for 'checkout', or, for actions
other than 'checkout' and 'bot-cycle', name configurations for which
compilers or glibc are to be built.

The 'list-compilers' command prints the name of each available
compiler configuration, without building anything.  The 'list-glibcs'
command prints the name of each glibc compiler configuration, followed
by the space, followed by the name of the compiler configuration used
for building this glibc variant.

"""

import argparse
import datetime
import email.mime.text
import email.utils
import json
import os
import re
import shutil
import smtplib
import stat
import subprocess
import sys
import time
import urllib.request

try:
    subprocess.run
except:
    class _CompletedProcess:
        def __init__(self, args, returncode, stdout=None, stderr=None):
            self.args = args
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def _run(*popenargs, input=None, timeout=None, check=False, **kwargs):
        assert(timeout is None)
        with subprocess.Popen(*popenargs, **kwargs) as process:
            try:
                stdout, stderr = process.communicate(input)
            except:
                process.kill()
                process.wait()
                raise
            returncode = process.poll()
            if check and returncode:
                raise subprocess.CalledProcessError(returncode, popenargs)
        return _CompletedProcess(popenargs, returncode, stdout, stderr)

    subprocess.run = _run


class Context(object):
    """The global state associated with builds in a given directory."""

    def __init__(self, topdir, parallelism, keep, replace_sources, strip,
                 full_gcc, action, shallow=False):
        """Initialize the context."""
        self.topdir = topdir
        self.parallelism = parallelism
        self.keep = keep
        self.replace_sources = replace_sources
        self.strip = strip
        self.full_gcc = full_gcc
        self.shallow = shallow
        self.srcdir = os.path.join(topdir, 'src')
        self.versions_json = os.path.join(self.srcdir, 'versions.json')
        self.build_state_json = os.path.join(topdir, 'build-state.json')
        self.bot_config_json = os.path.join(topdir, 'bot-config.json')
        self.installdir = os.path.join(topdir, 'install')
        self.host_libraries_installdir = os.path.join(self.installdir,
                                                      'host-libraries')
        self.builddir = os.path.join(topdir, 'build')
        self.logsdir = os.path.join(topdir, 'logs')
        self.logsdir_old = os.path.join(topdir, 'logs-old')
        self.makefile = os.path.join(self.builddir, 'Makefile')
        self.wrapper = os.path.join(self.builddir, 'wrapper')
        self.save_logs = os.path.join(self.builddir, 'save-logs')
        self.script_text = self.get_script_text()
        if action not in ('checkout', 'list-compilers', 'list-glibcs'):
            self.build_triplet = self.get_build_triplet()
            self.glibc_version = self.get_glibc_version()
        self.configs = {}
        self.glibc_configs = {}
        self.makefile_pieces = ['.PHONY: all\n']
        self.add_all_configs()
        self.load_versions_json()
        self.load_build_state_json()
        self.status_log_list = []
        self.email_warning = False

    def get_script_text(self):
        """Return the text of this script."""
        with open(sys.argv[0], 'r') as f:
            return f.read()

    def exec_self(self):
        """Re-execute this script with the same arguments."""
        sys.stdout.flush()
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def get_build_triplet(self):
        """Determine the build triplet with config.guess."""
        config_guess = os.path.join(self.component_srcdir('gcc'),
                                    'config.guess')
        cg_out = subprocess.run([config_guess], stdout=subprocess.PIPE,
                                check=True, universal_newlines=True).stdout
        return cg_out.rstrip()

    def get_glibc_version(self):
        """Determine the glibc version number (major.minor)."""
        version_h = os.path.join(self.component_srcdir('glibc'), 'version.h')
        with open(version_h, 'r') as f:
            lines = f.readlines()
        starttext = '#define VERSION "'
        for l in lines:
            if l.startswith(starttext):
                l = l[len(starttext):]
                l = l.rstrip('"\n')
                m = re.fullmatch('([0-9]+)\.([0-9]+)[.0-9]*', l)
                return '%s.%s' % m.group(1, 2)
        print('error: could not determine glibc version')
        exit(1)

    def add_all_configs(self):
        """Add all known glibc build configurations."""
        self.add_config(arch='aarch64',
                        os_name='linux-gnu',
                        extra_glibcs=[{'variant': 'disable-multi-arch',
                                       'cfg': ['--disable-multi-arch']}])
        self.add_config(arch='aarch64_be',
                        os_name='linux-gnu')
        self.add_config(arch='arc',
                        os_name='linux-gnu',
                        gcc_cfg=['--disable-multilib', '--with-cpu=hs38'])
        self.add_config(arch='arc',
                        os_name='linux-gnuhf',
                        gcc_cfg=['--disable-multilib', '--with-cpu=hs38_linux'])
        self.add_config(arch='arceb',
                        os_name='linux-gnu',
                        gcc_cfg=['--disable-multilib', '--with-cpu=hs38'])
        self.add_config(arch='alpha',
                        os_name='linux-gnu')
        self.add_config(arch='arm',
                        os_name='linux-gnueabi',
                        extra_glibcs=[{'variant': 'v4t',
                                       'ccopts': '-march=armv4t'}])
        self.add_config(arch='armeb',
                        os_name='linux-gnueabi')
        self.add_config(arch='armeb',
                        os_name='linux-gnueabi',
                        variant='be8',
                        gcc_cfg=['--with-arch=armv7-a'])
        self.add_config(arch='arm',
                        os_name='linux-gnueabihf',
                        gcc_cfg=['--with-float=hard', '--with-cpu=arm926ej-s'],
                        extra_glibcs=[{'variant': 'v7a',
                                       'ccopts': '-march=armv7-a -mfpu=vfpv3'},
                                      {'variant': 'thumb',
                                       'ccopts':
                                       '-mthumb -march=armv7-a -mfpu=vfpv3'},
                                      {'variant': 'v7a-disable-multi-arch',
                                       'ccopts': '-march=armv7-a -mfpu=vfpv3',
                                       'cfg': ['--disable-multi-arch']}])
        self.add_config(arch='armeb',
                        os_name='linux-gnueabihf',
                        gcc_cfg=['--with-float=hard', '--with-cpu=arm926ej-s'])
        self.add_config(arch='armeb',
                        os_name='linux-gnueabihf',
                        variant='be8',
                        gcc_cfg=['--with-float=hard', '--with-arch=armv7-a',
                                 '--with-fpu=vfpv3'])
        self.add_config(arch='csky',
                        os_name='linux-gnuabiv2',
                        variant='soft',
                        gcc_cfg=['--disable-multilib'])
        self.add_config(arch='csky',
                        os_name='linux-gnuabiv2',
                        gcc_cfg=['--with-float=hard', '--disable-multilib'])
        self.add_config(arch='hppa',
                        os_name='linux-gnu')
        self.add_config(arch='i686',
                        os_name='gnu')
        self.add_config(arch='ia64',
                        os_name='linux-gnu',
                        first_gcc_cfg=['--with-system-libunwind'],
                        binutils_cfg=['--enable-obsolete'])
        self.add_config(arch='m68k',
                        os_name='linux-gnu',
                        gcc_cfg=['--disable-multilib'])
        self.add_config(arch='m68k',
                        os_name='linux-gnu',
                        variant='coldfire',
                        gcc_cfg=['--with-arch=cf', '--disable-multilib'])
        self.add_config(arch='m68k',
                        os_name='linux-gnu',
                        variant='coldfire-soft',
                        gcc_cfg=['--with-arch=cf', '--with-cpu=54455',
                                 '--disable-multilib'])
        self.add_config(arch='microblaze',
                        os_name='linux-gnu',
                        gcc_cfg=['--disable-multilib'])
        self.add_config(arch='microblazeel',
                        os_name='linux-gnu',
                        gcc_cfg=['--disable-multilib'])
        self.add_config(arch='mips64',
                        os_name='linux-gnu',
                        gcc_cfg=['--with-mips-plt'],
                        glibcs=[{'variant': 'n32'},
                                {'arch': 'mips',
                                 'ccopts': '-mabi=32'},
                                {'variant': 'n64',
                                 'ccopts': '-mabi=64'}])
        self.add_config(arch='mips64',
                        os_name='linux-gnu',
                        variant='soft',
                        gcc_cfg=['--with-mips-plt', '--with-float=soft'],
                        glibcs=[{'variant': 'n32-soft'},
                                {'variant': 'soft',
                                 'arch': 'mips',
                                 'ccopts': '-mabi=32'},
                                {'variant': 'n64-soft',
                                 'ccopts': '-mabi=64'}])
        self.add_config(arch='mips64',
                        os_name='linux-gnu',
                        variant='nan2008',
                        gcc_cfg=['--with-mips-plt', '--with-nan=2008',
                                 '--with-arch-64=mips64r2',
                                 '--with-arch-32=mips32r2'],
                        glibcs=[{'variant': 'n32-nan2008'},
                                {'variant': 'nan2008',
                                 'arch': 'mips',
                                 'ccopts': '-mabi=32'},
                                {'variant': 'n64-nan2008',
                                 'ccopts': '-mabi=64'}])
        self.add_config(arch='mips64',
                        os_name='linux-gnu',
                        variant='nan2008-soft',
                        gcc_cfg=['--with-mips-plt', '--with-nan=2008',
                                 '--with-arch-64=mips64r2',
                                 '--with-arch-32=mips32r2',
                                 '--with-float=soft'],
                        glibcs=[{'variant': 'n32-nan2008-soft'},
                                {'variant': 'nan2008-soft',
                                 'arch': 'mips',
                                 'ccopts': '-mabi=32'},
                                {'variant': 'n64-nan2008-soft',
                                 'ccopts': '-mabi=64'}])
        self.add_config(arch='mips64el',
                        os_name='linux-gnu',
                        gcc_cfg=['--with-mips-plt'],
                        glibcs=[{'variant': 'n32'},
                                {'arch': 'mipsel',
                                 'ccopts': '-mabi=32'},
                                {'variant': 'n64',
                                 'ccopts': '-mabi=64'}])
        self.add_config(arch='mips64el',
                        os_name='linux-gnu',
                        variant='soft',
                        gcc_cfg=['--with-mips-plt', '--with-float=soft'],
                        glibcs=[{'variant': 'n32-soft'},
                                {'variant': 'soft',
                                 'arch': 'mipsel',
                                 'ccopts': '-mabi=32'},
                                {'variant': 'n64-soft',
                                 'ccopts': '-mabi=64'}])
        self.add_config(arch='mips64el',
                        os_name='linux-gnu',
                        variant='nan2008',
                        gcc_cfg=['--with-mips-plt', '--with-nan=2008',
                                 '--with-arch-64=mips64r2',
                                 '--with-arch-32=mips32r2'],
                        glibcs=[{'variant': 'n32-nan2008'},
                                {'variant': 'nan2008',
                                 'arch': 'mipsel',
                                 'ccopts': '-mabi=32'},
                                {'variant': 'n64-nan2008',
                                 'ccopts': '-mabi=64'}])
        self.add_config(arch='mips64el',
                        os_name='linux-gnu',
                        variant='nan2008-soft',
                        gcc_cfg=['--with-mips-plt', '--with-nan=2008',
                                 '--with-arch-64=mips64r2',
                                 '--with-arch-32=mips32r2',
                                 '--with-float=soft'],
                        glibcs=[{'variant': 'n32-nan2008-soft'},
                                {'variant': 'nan2008-soft',
                                 'arch': 'mipsel',
                                 'ccopts': '-mabi=32'},
                                {'variant': 'n64-nan2008-soft',
                                 'ccopts': '-mabi=64'}])
        self.add_config(arch='mipsisa64r6el',
                        os_name='linux-gnu',
                        gcc_cfg=['--with-mips-plt', '--with-nan=2008',
                                 '--with-arch-64=mips64r6',
                                 '--with-arch-32=mips32r6',
                                 '--with-float=hard'],
                        glibcs=[{'variant': 'n32'},
                                {'arch': 'mipsisa32r6el',
                                 'ccopts': '-mabi=32'},
                                {'variant': 'n64',
                                 'ccopts': '-mabi=64'}])
        self.add_config(arch='nios2',
                        os_name='linux-gnu')
        self.add_config(arch='powerpc',
                        os_name='linux-gnu',
                        gcc_cfg=['--disable-multilib', '--enable-secureplt'],
                        extra_glibcs=[{'variant': 'power4',
                                       'ccopts': '-mcpu=power4',
                                       'cfg': ['--with-cpu=power4']}])
        self.add_config(arch='powerpc',
                        os_name='linux-gnu',
                        variant='soft',
                        gcc_cfg=['--disable-multilib', '--with-float=soft',
                                 '--enable-secureplt'])
        self.add_config(arch='powerpc64',
                        os_name='linux-gnu',
                        gcc_cfg=['--disable-multilib', '--enable-secureplt'])
        self.add_config(arch='powerpc64le',
                        os_name='linux-gnu',
                        gcc_cfg=['--disable-multilib', '--enable-secureplt'])
        self.add_config(arch='riscv32',
                        os_name='linux-gnu',
                        variant='rv32imac-ilp32',
                        gcc_cfg=['--with-arch=rv32imac', '--with-abi=ilp32',
                                 '--disable-multilib'])
        self.add_config(arch='riscv32',
                        os_name='linux-gnu',
                        variant='rv32imafdc-ilp32',
                        gcc_cfg=['--with-arch=rv32imafdc', '--with-abi=ilp32',
                                 '--disable-multilib'])
        self.add_config(arch='riscv32',
                        os_name='linux-gnu',
                        variant='rv32imafdc-ilp32d',
                        gcc_cfg=['--with-arch=rv32imafdc', '--with-abi=ilp32d',
                                 '--disable-multilib'])
        self.add_config(arch='riscv64',
                        os_name='linux-gnu',
                        variant='rv64imac-lp64',
                        gcc_cfg=['--with-arch=rv64imac', '--with-abi=lp64',
                                 '--disable-multilib'])
        self.add_config(arch='riscv64',
                        os_name='linux-gnu',
                        variant='rv64imafdc-lp64',
                        gcc_cfg=['--with-arch=rv64imafdc', '--with-abi=lp64',
                                 '--disable-multilib'])
        self.add_config(arch='riscv64',
                        os_name='linux-gnu',
                        variant='rv64imafdc-lp64d',
                        gcc_cfg=['--with-arch=rv64imafdc', '--with-abi=lp64d',
                                 '--disable-multilib'])
        self.add_config(arch='s390x',
                        os_name='linux-gnu',
                        glibcs=[{},
                                {'arch': 's390', 'ccopts': '-m31'}],
                        extra_glibcs=[{'variant': 'O3',
                                       'cflags': '-O3'}])
        self.add_config(arch='sh3',
                        os_name='linux-gnu')
        self.add_config(arch='sh3eb',
                        os_name='linux-gnu')
        self.add_config(arch='sh4',
                        os_name='linux-gnu')
        self.add_config(arch='sh4eb',
                        os_name='linux-gnu')
        self.add_config(arch='sh4',
                        os_name='linux-gnu',
                        variant='soft',
                        gcc_cfg=['--without-fp'])
        self.add_config(arch='sh4eb',
                        os_name='linux-gnu',
                        variant='soft',
                        gcc_cfg=['--without-fp'])
        self.add_config(arch='sparc64',
                        os_name='linux-gnu',
                        glibcs=[{},
                                {'arch': 'sparcv9',
                                 'ccopts': '-m32 -mlong-double-128 -mcpu=v9'}],
                        extra_glibcs=[{'variant': 'leon3',
                                       'arch' : 'sparcv8',
                                       'ccopts' : '-m32 -mlong-double-128 -mcpu=leon3'},
                                      {'variant': 'disable-multi-arch',
                                       'cfg': ['--disable-multi-arch']},
                                      {'variant': 'disable-multi-arch',
                                       'arch': 'sparcv9',
                                       'ccopts': '-m32 -mlong-double-128 -mcpu=v9',
                                       'cfg': ['--disable-multi-arch']}])
        self.add_config(arch='x86_64',
                        os_name='linux-gnu',
                        gcc_cfg=['--with-multilib-list=m64,m32,mx32'],
                        glibcs=[{},
                                {'variant': 'x32', 'ccopts': '-mx32'},
                                {'arch': 'i686', 'ccopts': '-m32 -march=i686'}],
                        extra_glibcs=[{'variant': 'disable-multi-arch',
                                       'cfg': ['--disable-multi-arch']},
                                      {'variant': 'minimal',
                                       'cfg': ['--disable-multi-arch',
                                               '--disable-profile',
                                               '--disable-timezone-tools',
                                               '--disable-mathvec',
                                               '--disable-tunables',
                                               '--disable-crypt',
                                               '--disable-experimental-malloc',
                                               '--disable-build-nscd',
                                               '--disable-nscd']},
                                      {'variant': 'static-pie',
                                       'cfg': ['--enable-static-pie']},
                                      {'variant': 'x32-static-pie',
                                       'ccopts': '-mx32',
                                       'cfg': ['--enable-static-pie']},
                                      {'variant': 'static-pie',
                                       'arch': 'i686',
                                       'ccopts': '-m32 -march=i686',
                                       'cfg': ['--enable-static-pie']},
                                      {'variant': 'disable-multi-arch',
                                       'arch': 'i686',
                                       'ccopts': '-m32 -march=i686',
                                       'cfg': ['--disable-multi-arch']},
                                      {'arch': 'i486',
                                       'ccopts': '-m32 -march=i486'},
                                      {'arch': 'i586',
                                       'ccopts': '-m32 -march=i586'}])

    def add_config(self, **args):
        """Add an individual build configuration."""
        cfg = Config(self, **args)
        if cfg.name in self.configs:
            print('error: duplicate config %s' % cfg.name)
            exit(1)
        self.configs[cfg.name] = cfg
        for c in cfg.all_glibcs:
            if c.name in self.glibc_configs:
                print('error: duplicate glibc config %s' % c.name)
                exit(1)
            self.glibc_configs[c.name] = c

    def component_srcdir(self, component):
        """Return the source directory for a given component, e.g. gcc."""
        return os.path.join(self.srcdir, component)

    def component_builddir(self, action, config, component, subconfig=None):
        """Return the directory to use for a build."""
        if config is None:
            # Host libraries.
            assert subconfig is None
            return os.path.join(self.builddir, action, component)
        if subconfig is None:
            return os.path.join(self.builddir, action, config, component)
        else:
            # glibc build as part of compiler build.
            return os.path.join(self.builddir, action, config, component,
                                subconfig)

    def compiler_installdir(self, config):
        """Return the directory in which to install a compiler."""
        return os.path.join(self.installdir, 'compilers', config)

    def compiler_bindir(self, config):
        """Return the directory in which to find compiler binaries."""
        return os.path.join(self.compiler_installdir(config), 'bin')

    def compiler_sysroot(self, config):
        """Return the sysroot directory for a compiler."""
        return os.path.join(self.compiler_installdir(config), 'sysroot')

    def glibc_installdir(self, config):
        """Return the directory in which to install glibc."""
        return os.path.join(self.installdir, 'glibcs', config)

    def run_builds(self, action, configs):
        """Run the requested builds."""
        if action == 'checkout':
            self.checkout(configs)
            return
        if action == 'bot-cycle':
            if configs:
                print('error: configurations specified for bot-cycle')
                exit(1)
            self.bot_cycle()
            return
        if action == 'bot':
            if configs:
                print('error: configurations specified for bot')
                exit(1)
            self.bot()
            return
        if action in ('host-libraries', 'list-compilers',
                      'list-glibcs') and configs:
            print('error: configurations specified for ' + action)
            exit(1)
        if action == 'list-compilers':
            for name in sorted(self.configs.keys()):
                print(name)
            return
        if action == 'list-glibcs':
            for config in sorted(self.glibc_configs.values(),
                                 key=lambda c: c.name):
                print(config.name, config.compiler.name)
            return
        self.clear_last_build_state(action)
        build_time = datetime.datetime.utcnow()
        if action == 'host-libraries':
            build_components = ('gmp', 'mpfr', 'mpc')
            old_components = ()
            old_versions = {}
            self.build_host_libraries()
        elif action == 'compilers':
            build_components = ('binutils', 'gcc', 'glibc', 'linux', 'mig',
                                'gnumach', 'hurd')
            old_components = ('gmp', 'mpfr', 'mpc')
            old_versions = self.build_state['host-libraries']['build-versions']
            self.build_compilers(configs)
        else:
            build_components = ('glibc',)
            old_components = ('gmp', 'mpfr', 'mpc', 'binutils', 'gcc', 'linux',
                              'mig', 'gnumach', 'hurd')
            old_versions = self.build_state['compilers']['build-versions']
            if action == 'update-syscalls':
                self.update_syscalls(configs)
            else:
                self.build_glibcs(configs)
        self.write_files()
        self.do_build()
        if configs:
            # Partial build, do not update stored state.
            return
        build_versions = {}
        for k in build_components:
            if k in self.versions:
                build_versions[k] = {'version': self.versions[k]['version'],
                                     'revision': self.versions[k]['revision']}
        for k in old_components:
            if k in old_versions:
                build_versions[k] = {'version': old_versions[k]['version'],
                                     'revision': old_versions[k]['revision']}
        self.update_build_state(action, build_time, build_versions)

    @staticmethod
    def remove_dirs(*args):
        """Remove directories and their contents if they exist."""
        for dir in args:
            shutil.rmtree(dir, ignore_errors=True)

    @staticmethod
    def remove_recreate_dirs(*args):
        """Remove directories if they exist, and create them as empty."""
        Context.remove_dirs(*args)
        for dir in args:
            os.makedirs(dir, exist_ok=True)

    def add_makefile_cmdlist(self, target, cmdlist, logsdir):
        """Add makefile text for a list of commands."""
        commands = cmdlist.makefile_commands(self.wrapper, logsdir)
        self.makefile_pieces.append('all: %s\n.PHONY: %s\n%s:\n%s\n' %
                                    (target, target, target, commands))
        self.status_log_list.extend(cmdlist.status_logs(logsdir))

    def write_files(self):
        """Write out the Makefile and wrapper script."""
        mftext = ''.join(self.makefile_pieces)
        with open(self.makefile, 'w') as f:
            f.write(mftext)
        wrapper_text = (
            '#!/bin/sh\n'
            'prev_base=$1\n'
            'this_base=$2\n'
            'desc=$3\n'
            'dir=$4\n'
            'path=$5\n'
            'shift 5\n'
            'prev_status=$prev_base-status.txt\n'
            'this_status=$this_base-status.txt\n'
            'this_log=$this_base-log.txt\n'
            'date > "$this_log"\n'
            'echo >> "$this_log"\n'
            'echo "Description: $desc" >> "$this_log"\n'
            'printf "%s" "Command:" >> "$this_log"\n'
            'for word in "$@"; do\n'
            '  if expr "$word" : "[]+,./0-9@A-Z_a-z-]\\\\{1,\\\\}\\$" > /dev/null; then\n'
            '    printf " %s" "$word"\n'
            '  else\n'
            '    printf " \'"\n'
            '    printf "%s" "$word" | sed -e "s/\'/\'\\\\\\\\\'\'/"\n'
            '    printf "\'"\n'
            '  fi\n'
            'done >> "$this_log"\n'
            'echo >> "$this_log"\n'
            'echo "Directory: $dir" >> "$this_log"\n'
            'echo "Path addition: $path" >> "$this_log"\n'
            'echo >> "$this_log"\n'
            'record_status ()\n'
            '{\n'
            '  echo >> "$this_log"\n'
            '  echo "$1: $desc" > "$this_status"\n'
            '  echo "$1: $desc" >> "$this_log"\n'
            '  echo >> "$this_log"\n'
            '  date >> "$this_log"\n'
            '  echo "$1: $desc"\n'
            '  exit 0\n'
            '}\n'
            'check_error ()\n'
            '{\n'
            '  if [ "$1" != "0" ]; then\n'
            '    record_status FAIL\n'
            '  fi\n'
            '}\n'
            'if [ "$prev_base" ] && ! grep -q "^PASS" "$prev_status"; then\n'
            '    record_status UNRESOLVED\n'
            'fi\n'
            'if [ "$dir" ]; then\n'
            '  cd "$dir"\n'
            '  check_error "$?"\n'
            'fi\n'
            'if [ "$path" ]; then\n'
            '  PATH=$path:$PATH\n'
            'fi\n'
            '"$@" < /dev/null >> "$this_log" 2>&1\n'
            'check_error "$?"\n'
            'record_status PASS\n')
        with open(self.wrapper, 'w') as f:
            f.write(wrapper_text)
        # Mode 0o755.
        mode_exec = (stat.S_IRWXU|stat.S_IRGRP|stat.S_IXGRP|
                     stat.S_IROTH|stat.S_IXOTH)
        os.chmod(self.wrapper, mode_exec)
        save_logs_text = (
            '#!/bin/sh\n'
            'if ! [ -f tests.sum ]; then\n'
            '  echo "No test summary available."\n'
            '  exit 0\n'
            'fi\n'
            'save_file ()\n'
            '{\n'
            '  echo "Contents of $1:"\n'
            '  echo\n'
            '  cat "$1"\n'
            '  echo\n'
            '  echo "End of contents of $1."\n'
            '  echo\n'
            '}\n'
            'save_file tests.sum\n'
            'non_pass_tests=$(grep -v "^PASS: " tests.sum | sed -e "s/^PASS: //")\n'
            'for t in $non_pass_tests; do\n'
            '  if [ -f "$t.out" ]; then\n'
            '    save_file "$t.out"\n'
            '  fi\n'
            'done\n')
        with open(self.save_logs, 'w') as f:
            f.write(save_logs_text)
        os.chmod(self.save_logs, mode_exec)

    def do_build(self):
        """Do the actual build."""
        cmd = ['make', '-O', '-j%d' % self.parallelism]
        subprocess.run(cmd, cwd=self.builddir, check=True)

    def build_host_libraries(self):
        """Build the host libraries."""
        installdir = self.host_libraries_installdir
        builddir = os.path.join(self.builddir, 'host-libraries')
        logsdir = os.path.join(self.logsdir, 'host-libraries')
        self.remove_recreate_dirs(installdir, builddir, logsdir)
        cmdlist = CommandList('host-libraries', self.keep)
        self.build_host_library(cmdlist, 'gmp')
        self.build_host_library(cmdlist, 'mpfr',
                                ['--with-gmp=%s' % installdir])
        self.build_host_library(cmdlist, 'mpc',
                                ['--with-gmp=%s' % installdir,
                                '--with-mpfr=%s' % installdir])
        cmdlist.add_command('done', ['touch', os.path.join(installdir, 'ok')])
        self.add_makefile_cmdlist('host-libraries', cmdlist, logsdir)

    def build_host_library(self, cmdlist, lib, extra_opts=None):
        """Build one host library."""
        srcdir = self.component_srcdir(lib)
        builddir = self.component_builddir('host-libraries', None, lib)
        installdir = self.host_libraries_installdir
        cmdlist.push_subdesc(lib)
        cmdlist.create_use_dir(builddir)
        cfg_cmd = [os.path.join(srcdir, 'configure'),
                   '--prefix=%s' % installdir,
                   '--disable-shared']
        if extra_opts:
            cfg_cmd.extend (extra_opts)
        cmdlist.add_command('configure', cfg_cmd)
        cmdlist.add_command('build', ['make'])
        cmdlist.add_command('check', ['make', 'check'])
        cmdlist.add_command('install', ['make', 'install'])
        cmdlist.cleanup_dir()
        cmdlist.pop_subdesc()

    def build_compilers(self, configs):
        """Build the compilers."""
        if not configs:
            self.remove_dirs(os.path.join(self.builddir, 'compilers'))
            self.remove_dirs(os.path.join(self.installdir, 'compilers'))
            self.remove_dirs(os.path.join(self.logsdir, 'compilers'))
            configs = sorted(self.configs.keys())
        for c in configs:
            self.configs[c].build()

    def build_glibcs(self, configs):
        """Build the glibcs."""
        if not configs:
            self.remove_dirs(os.path.join(self.builddir, 'glibcs'))
            self.remove_dirs(os.path.join(self.installdir, 'glibcs'))
            self.remove_dirs(os.path.join(self.logsdir, 'glibcs'))
            configs = sorted(self.glibc_configs.keys())
        for c in configs:
            self.glibc_configs[c].build()

    def update_syscalls(self, configs):
        """Update the glibc syscall lists."""
        if not configs:
            self.remove_dirs(os.path.join(self.builddir, 'update-syscalls'))
            self.remove_dirs(os.path.join(self.logsdir, 'update-syscalls'))
            configs = sorted(self.glibc_configs.keys())
        for c in configs:
            self.glibc_configs[c].update_syscalls()

    def load_versions_json(self):
        """Load information about source directory versions."""
        if not os.access(self.versions_json, os.F_OK):
            self.versions = {}
            return
        with open(self.versions_json, 'r') as f:
            self.versions = json.load(f)

    def store_json(self, data, filename):
        """Store information in a JSON file."""
        filename_tmp = filename + '.tmp'
        with open(filename_tmp, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.rename(filename_tmp, filename)

    def store_versions_json(self):
        """Store information about source directory versions."""
        self.store_json(self.versions, self.versions_json)

    def set_component_version(self, component, version, explicit, revision):
        """Set the version information for a component."""
        self.versions[component] = {'version': version,
                                    'explicit': explicit,
                                    'revision': revision}
        self.store_versions_json()

    def checkout(self, versions):
        """Check out the desired component versions."""
        default_versions = {'binutils': 'vcs-2.36',
                            'gcc': 'vcs-11',
                            'glibc': 'vcs-mainline',
                            'gmp': '6.2.1',
                            'linux': '5.13',
                            'mpc': '1.2.1',
                            'mpfr': '4.1.0',
                            'mig': 'vcs-mainline',
                            'gnumach': 'vcs-mainline',
                            'hurd': 'vcs-mainline'}
        use_versions = {}
        explicit_versions = {}
        for v in versions:
            found_v = False
            for k in default_versions.keys():
                kx = k + '-'
                if v.startswith(kx):
                    vx = v[len(kx):]
                    if k in use_versions:
                        print('error: multiple versions for %s' % k)
                        exit(1)
                    use_versions[k] = vx
                    explicit_versions[k] = True
                    found_v = True
                    break
            if not found_v:
                print('error: unknown component in %s' % v)
                exit(1)
        for k in default_versions.keys():
            if k not in use_versions:
                if k in self.versions and self.versions[k]['explicit']:
                    use_versions[k] = self.versions[k]['version']
                    explicit_versions[k] = True
                else:
                    use_versions[k] = default_versions[k]
                    explicit_versions[k] = False
        os.makedirs(self.srcdir, exist_ok=True)
        for k in sorted(default_versions.keys()):
            update = os.access(self.component_srcdir(k), os.F_OK)
            v = use_versions[k]
            if (update and
                k in self.versions and
                v != self.versions[k]['version']):
                if not self.replace_sources:
                    print('error: version of %s has changed from %s to %s, '
                          'use --replace-sources to check out again' %
                          (k, self.versions[k]['version'], v))
                    exit(1)
                shutil.rmtree(self.component_srcdir(k))
                update = False
            if v.startswith('vcs-'):
                revision = self.checkout_vcs(k, v[4:], update)
            else:
                self.checkout_tar(k, v, update)
                revision = v
            self.set_component_version(k, v, explicit_versions[k], revision)
        if self.get_script_text() != self.script_text:
            # Rerun the checkout process in case the updated script
            # uses different default versions or new components.
            self.exec_self()

    def checkout_vcs(self, component, version, update):
        """Check out the given version of the given component from version
        control.  Return a revision identifier."""
        if component == 'binutils':
            git_url = 'git://sourceware.org/git/binutils-gdb.git'
            if version == 'mainline':
                git_branch = 'master'
            else:
                trans = str.maketrans({'.': '_'})
                git_branch = 'binutils-%s-branch' % version.translate(trans)
            return self.git_checkout(component, git_url, git_branch, update)
        elif component == 'gcc':
            if version == 'mainline':
                branch = 'master'
            else:
                branch = 'releases/gcc-%s' % version
            return self.gcc_checkout(branch, update)
        elif component == 'glibc':
            git_url = 'git://sourceware.org/git/glibc.git'
            if version == 'mainline':
                git_branch = 'master'
            else:
                git_branch = 'release/%s/master' % version
            r = self.git_checkout(component, git_url, git_branch, update)
            self.fix_glibc_timestamps()
            return r
        elif component == 'gnumach':
            git_url = 'git://git.savannah.gnu.org/hurd/gnumach.git'
            git_branch = 'master'
            r = self.git_checkout(component, git_url, git_branch, update)
            subprocess.run(['autoreconf', '-i'],
                           cwd=self.component_srcdir(component), check=True)
            return r
        elif component == 'mig':
            git_url = 'git://git.savannah.gnu.org/hurd/mig.git'
            git_branch = 'master'
            r = self.git_checkout(component, git_url, git_branch, update)
            subprocess.run(['autoreconf', '-i'],
                           cwd=self.component_srcdir(component), check=True)
            return r
        elif component == 'hurd':
            git_url = 'git://git.savannah.gnu.org/hurd/hurd.git'
            git_branch = 'master'
            r = self.git_checkout(component, git_url, git_branch, update)
            subprocess.run(['autoconf'],
                           cwd=self.component_srcdir(component), check=True)
            return r
        else:
            print('error: component %s coming from VCS' % component)
            exit(1)

    def git_checkout(self, component, git_url, git_branch, update):
        """Check out a component from git.  Return a commit identifier."""
        if update:
            subprocess.run(['git', 'remote', 'prune', 'origin'],
                           cwd=self.component_srcdir(component), check=True)
            if self.replace_sources:
                subprocess.run(['git', 'clean', '-dxfq'],
                               cwd=self.component_srcdir(component), check=True)
            subprocess.run(['git', 'pull', '-q'],
                           cwd=self.component_srcdir(component), check=True)
        else:
            if self.shallow:
                depth_arg = ('--depth', '1')
            else:
                depth_arg = ()
            subprocess.run(['git', 'clone', '-q', '-b', git_branch,
                            *depth_arg, git_url,
                            self.component_srcdir(component)], check=True)
        r = subprocess.run(['git', 'rev-parse', 'HEAD'],
                           cwd=self.component_srcdir(component),
                           stdout=subprocess.PIPE,
                           check=True, universal_newlines=True).stdout
        return r.rstrip()

    def fix_glibc_timestamps(self):
        """Fix timestamps in a glibc checkout."""
        # Ensure that builds do not try to regenerate generated files
        # in the source tree.
        srcdir = self.component_srcdir('glibc')
        # These files have Makefile dependencies to regenerate them in
        # the source tree that may be active during a normal build.
        # Some other files have such dependencies but do not need to
        # be touched because nothing in a build depends on the files
        # in question.
        for f in ('sysdeps/mach/hurd/bits/errno.h',):
            to_touch = os.path.join(srcdir, f)
            subprocess.run(['touch', '-c', to_touch], check=True)
        for dirpath, dirnames, filenames in os.walk(srcdir):
            for f in filenames:
                if (f == 'configure' or
                    f == 'preconfigure' or
                    f.endswith('-kw.h')):
                    to_touch = os.path.join(dirpath, f)
                    subprocess.run(['touch', to_touch], check=True)

    def gcc_checkout(self, branch, update):
        """Check out GCC from git.  Return the commit identifier."""
        if os.access(os.path.join(self.component_srcdir('gcc'), '.svn'),
                     os.F_OK):
            if not self.replace_sources:
                print('error: GCC has moved from SVN to git, use '
                      '--replace-sources to check out again')
                exit(1)
            shutil.rmtree(self.component_srcdir('gcc'))
            update = False
        if not update:
            self.git_checkout('gcc', 'git://gcc.gnu.org/git/gcc.git',
                              branch, update)
        subprocess.run(['contrib/gcc_update', '--silent'],
                       cwd=self.component_srcdir('gcc'), check=True)
        r = subprocess.run(['git', 'rev-parse', 'HEAD'],
                           cwd=self.component_srcdir('gcc'),
                           stdout=subprocess.PIPE,
                           check=True, universal_newlines=True).stdout
        return r.rstrip()

    def checkout_tar(self, component, version, update):
        """Check out the given version of the given component from a
        tarball."""
        if update:
            return
        url_map = {'binutils': 'https://ftp.gnu.org/gnu/binutils/binutils-%(version)s.tar.bz2',
                   'gcc': 'https://ftp.gnu.org/gnu/gcc/gcc-%(version)s/gcc-%(version)s.tar.gz',
                   'gmp': 'https://ftp.gnu.org/gnu/gmp/gmp-%(version)s.tar.xz',
                   'linux': 'https://www.kernel.org/pub/linux/kernel/v%(major)s.x/linux-%(version)s.tar.xz',
                   'mpc': 'https://ftp.gnu.org/gnu/mpc/mpc-%(version)s.tar.gz',
                   'mpfr': 'https://ftp.gnu.org/gnu/mpfr/mpfr-%(version)s.tar.xz',
                   'mig': 'https://ftp.gnu.org/gnu/mig/mig-%(version)s.tar.bz2',
                   'gnumach': 'https://ftp.gnu.org/gnu/gnumach/gnumach-%(version)s.tar.bz2',
                   'hurd': 'https://ftp.gnu.org/gnu/hurd/hurd-%(version)s.tar.bz2'}
        if component not in url_map:
            print('error: component %s coming from tarball' % component)
            exit(1)
        version_major = version.split('.')[0]
        url = url_map[component] % {'version': version, 'major': version_major}
        filename = os.path.join(self.srcdir, url.split('/')[-1])
        response = urllib.request.urlopen(url)
        data = response.read()
        with open(filename, 'wb') as f:
            f.write(data)
        subprocess.run(['tar', '-C', self.srcdir, '-x', '-f', filename],
                       check=True)
        os.rename(os.path.join(self.srcdir, '%s-%s' % (component, version)),
                  self.component_srcdir(component))
        os.remove(filename)

    def load_build_state_json(self):
        """Load information about the state of previous builds."""
        if os.access(self.build_state_json, os.F_OK):
            with open(self.build_state_json, 'r') as f:
                self.build_state = json.load(f)
        else:
            self.build_state = {}
        for k in ('host-libraries', 'compilers', 'glibcs', 'update-syscalls'):
            if k not in self.build_state:
                self.build_state[k] = {}
            if 'build-time' not in self.build_state[k]:
                self.build_state[k]['build-time'] = ''
            if 'build-versions' not in self.build_state[k]:
                self.build_state[k]['build-versions'] = {}
            if 'build-results' not in self.build_state[k]:
                self.build_state[k]['build-results'] = {}
            if 'result-changes' not in self.build_state[k]:
                self.build_state[k]['result-changes'] = {}
            if 'ever-passed' not in self.build_state[k]:
                self.build_state[k]['ever-passed'] = []

    def store_build_state_json(self):
        """Store information about the state of previous builds."""
        self.store_json(self.build_state, self.build_state_json)

    def clear_last_build_state(self, action):
        """Clear information about the state of part of the build."""
        # We clear the last build time and versions when starting a
        # new build.  The results of the last build are kept around,
        # as comparison is still meaningful if this build is aborted
        # and a new one started.
        self.build_state[action]['build-time'] = ''
        self.build_state[action]['build-versions'] = {}
        self.store_build_state_json()

    def update_build_state(self, action, build_time, build_versions):
        """Update the build state after a build."""
        build_time = build_time.replace(microsecond=0)
        self.build_state[action]['build-time'] = str(build_time)
        self.build_state[action]['build-versions'] = build_versions
        build_results = {}
        for log in self.status_log_list:
            with open(log, 'r') as f:
                log_text = f.read()
            log_text = log_text.rstrip()
            m = re.fullmatch('([A-Z]+): (.*)', log_text)
            result = m.group(1)
            test_name = m.group(2)
            assert test_name not in build_results
            build_results[test_name] = result
        old_build_results = self.build_state[action]['build-results']
        self.build_state[action]['build-results'] = build_results
        result_changes = {}
        all_tests = set(old_build_results.keys()) | set(build_results.keys())
        for t in all_tests:
            if t in old_build_results:
                old_res = old_build_results[t]
            else:
                old_res = '(New test)'
            if t in build_results:
                new_res = build_results[t]
            else:
                new_res = '(Test removed)'
            if old_res != new_res:
                result_changes[t] = '%s -> %s' % (old_res, new_res)
        self.build_state[action]['result-changes'] = result_changes
        old_ever_passed = {t for t in self.build_state[action]['ever-passed']
                           if t in build_results}
        new_passes = {t for t in build_results if build_results[t] == 'PASS'}
        self.build_state[action]['ever-passed'] = sorted(old_ever_passed |
                                                         new_passes)
        self.store_build_state_json()

    def load_bot_config_json(self):
        """Load bot configuration."""
        with open(self.bot_config_json, 'r') as f:
            self.bot_config = json.load(f)

    def part_build_old(self, action, delay):
        """Return whether the last build for a given action was at least a
        given number of seconds ago, or does not have a time recorded."""
        old_time_str = self.build_state[action]['build-time']
        if not old_time_str:
            return True
        old_time = datetime.datetime.strptime(old_time_str,
                                              '%Y-%m-%d %H:%M:%S')
        new_time = datetime.datetime.utcnow()
        delta = new_time - old_time
        return delta.total_seconds() >= delay

    def bot_cycle(self):
        """Run a single round of checkout and builds."""
        print('Bot cycle starting %s.' % str(datetime.datetime.utcnow()))
        self.load_bot_config_json()
        actions = ('host-libraries', 'compilers', 'glibcs')
        self.bot_run_self(['--replace-sources'], 'checkout')
        self.load_versions_json()
        if self.get_script_text() != self.script_text:
            print('Script changed, re-execing.')
            # On script change, all parts of the build should be rerun.
            for a in actions:
                self.clear_last_build_state(a)
            self.exec_self()
        check_components = {'host-libraries': ('gmp', 'mpfr', 'mpc'),
                            'compilers': ('binutils', 'gcc', 'glibc', 'linux',
                                          'mig', 'gnumach', 'hurd'),
                            'glibcs': ('glibc',)}
        must_build = {}
        for a in actions:
            build_vers = self.build_state[a]['build-versions']
            must_build[a] = False
            if not self.build_state[a]['build-time']:
                must_build[a] = True
            old_vers = {}
            new_vers = {}
            for c in check_components[a]:
                if c in build_vers:
                    old_vers[c] = build_vers[c]
                new_vers[c] = {'version': self.versions[c]['version'],
                               'revision': self.versions[c]['revision']}
            if new_vers == old_vers:
                print('Versions for %s unchanged.' % a)
            else:
                print('Versions changed or rebuild forced for %s.' % a)
                if a == 'compilers' and not self.part_build_old(
                        a, self.bot_config['compilers-rebuild-delay']):
                    print('Not requiring rebuild of compilers this soon.')
                else:
                    must_build[a] = True
        if must_build['host-libraries']:
            must_build['compilers'] = True
        if must_build['compilers']:
            must_build['glibcs'] = True
        for a in actions:
            if must_build[a]:
                print('Must rebuild %s.' % a)
                self.clear_last_build_state(a)
            else:
                print('No need to rebuild %s.' % a)
        if os.access(self.logsdir, os.F_OK):
            shutil.rmtree(self.logsdir_old, ignore_errors=True)
            shutil.copytree(self.logsdir, self.logsdir_old)
        for a in actions:
            if must_build[a]:
                build_time = datetime.datetime.utcnow()
                print('Rebuilding %s at %s.' % (a, str(build_time)))
                self.bot_run_self([], a)
                self.load_build_state_json()
                self.bot_build_mail(a, build_time)
        print('Bot cycle done at %s.' % str(datetime.datetime.utcnow()))

    def bot_build_mail(self, action, build_time):
        """Send email with the results of a build."""
        if not ('email-from' in self.bot_config and
                'email-server' in self.bot_config and
                'email-subject' in self.bot_config and
                'email-to' in self.bot_config):
            if not self.email_warning:
                print("Email not configured, not sending.")
                self.email_warning = True
            return

        build_time = build_time.replace(microsecond=0)
        subject = (self.bot_config['email-subject'] %
                   {'action': action,
                    'build-time': str(build_time)})
        results = self.build_state[action]['build-results']
        changes = self.build_state[action]['result-changes']
        ever_passed = set(self.build_state[action]['ever-passed'])
        versions = self.build_state[action]['build-versions']
        new_regressions = {k for k in changes if changes[k] == 'PASS -> FAIL'}
        all_regressions = {k for k in ever_passed if results[k] == 'FAIL'}
        all_fails = {k for k in results if results[k] == 'FAIL'}
        if new_regressions:
            new_reg_list = sorted(['FAIL: %s' % k for k in new_regressions])
            new_reg_text = ('New regressions:\n\n%s\n\n' %
                            '\n'.join(new_reg_list))
        else:
            new_reg_text = ''
        if all_regressions:
            all_reg_list = sorted(['FAIL: %s' % k for k in all_regressions])
            all_reg_text = ('All regressions:\n\n%s\n\n' %
                            '\n'.join(all_reg_list))
        else:
            all_reg_text = ''
        if all_fails:
            all_fail_list = sorted(['FAIL: %s' % k for k in all_fails])
            all_fail_text = ('All failures:\n\n%s\n\n' %
                             '\n'.join(all_fail_list))
        else:
            all_fail_text = ''
        if changes:
            changes_list = sorted(changes.keys())
            changes_list = ['%s: %s' % (changes[k], k) for k in changes_list]
            changes_text = ('All changed results:\n\n%s\n\n' %
                            '\n'.join(changes_list))
        else:
            changes_text = ''
        results_text = (new_reg_text + all_reg_text + all_fail_text +
                        changes_text)
        if not results_text:
            results_text = 'Clean build with unchanged results.\n\n'
        versions_list = sorted(versions.keys())
        versions_list = ['%s: %s (%s)' % (k, versions[k]['version'],
                                          versions[k]['revision'])
                         for k in versions_list]
        versions_text = ('Component versions for this build:\n\n%s\n' %
                         '\n'.join(versions_list))
        body_text = results_text + versions_text
        msg = email.mime.text.MIMEText(body_text)
        msg['Subject'] = subject
        msg['From'] = self.bot_config['email-from']
        msg['To'] = self.bot_config['email-to']
        msg['Message-ID'] = email.utils.make_msgid()
        msg['Date'] = email.utils.format_datetime(datetime.datetime.utcnow())
        with smtplib.SMTP(self.bot_config['email-server']) as s:
            s.send_message(msg)

    def bot_run_self(self, opts, action, check=True):
        """Run a copy of this script with given options."""
        cmd = [sys.executable, sys.argv[0], '--keep=none',
               '-j%d' % self.parallelism]
        if self.full_gcc:
            cmd.append('--full-gcc')
        cmd.extend(opts)
        cmd.extend([self.topdir, action])
        sys.stdout.flush()
        subprocess.run(cmd, check=check)

    def bot(self):
        """Run repeated rounds of checkout and builds."""
        while True:
            self.load_bot_config_json()
            if not self.bot_config['run']:
                print('Bot exiting by request.')
                exit(0)
            self.bot_run_self([], 'bot-cycle', check=False)
            self.load_bot_config_json()
            if not self.bot_config['run']:
                print('Bot exiting by request.')
                exit(0)
            time.sleep(self.bot_config['delay'])
            if self.get_script_text() != self.script_text:
                print('Script changed, bot re-execing.')
                self.exec_self()

class LinuxHeadersPolicyForBuild(object):
    """Names and directories for installing Linux headers.  Build variant."""

    def __init__(self, config):
        self.arch = config.arch
        self.srcdir = config.ctx.component_srcdir('linux')
        self.builddir = config.component_builddir('linux')
        self.headers_dir = os.path.join(config.sysroot, 'usr')

class LinuxHeadersPolicyForUpdateSyscalls(object):
    """Names and directories for Linux headers.  update-syscalls variant."""

    def __init__(self, glibc, headers_dir):
        self.arch = glibc.compiler.arch
        self.srcdir = glibc.compiler.ctx.component_srcdir('linux')
        self.builddir = glibc.ctx.component_builddir(
            'update-syscalls', glibc.name, 'build-linux')
        self.headers_dir = headers_dir

def install_linux_headers(policy, cmdlist):
    """Install Linux kernel headers."""
    arch_map = {'aarch64': 'arm64',
                'alpha': 'alpha',
                'arc': 'arc',
                'arm': 'arm',
                'csky': 'csky',
                'hppa': 'parisc',
                'i486': 'x86',
                'i586': 'x86',
                'i686': 'x86',
                'i786': 'x86',
                'ia64': 'ia64',
                'm68k': 'm68k',
                'microblaze': 'microblaze',
                'mips': 'mips',
                'nios2': 'nios2',
                'powerpc': 'powerpc',
                's390': 's390',
                'riscv32': 'riscv',
                'riscv64': 'riscv',
                'sh': 'sh',
                'sparc': 'sparc',
                'x86_64': 'x86'}
    linux_arch = None
    for k in arch_map:
        if policy.arch.startswith(k):
            linux_arch = arch_map[k]
            break
    assert linux_arch is not None
    cmdlist.push_subdesc('linux')
    cmdlist.create_use_dir(policy.builddir)
    cmdlist.add_command('install-headers',
                        ['make', '-C', policy.srcdir, 'O=%s' % policy.builddir,
                         'ARCH=%s' % linux_arch,
                         'INSTALL_HDR_PATH=%s' % policy.headers_dir,
                         'headers_install'])
    cmdlist.cleanup_dir()
    cmdlist.pop_subdesc()

class Config(object):
    """A configuration for building a compiler and associated libraries."""

    def __init__(self, ctx, arch, os_name, variant=None, gcc_cfg=None,
                 first_gcc_cfg=None, binutils_cfg=None, glibcs=None,
                 extra_glibcs=None):
        """Initialize a Config object."""
        self.ctx = ctx
        self.arch = arch
        self.os = os_name
        self.variant = variant
        if variant is None:
            self.name = '%s-%s' % (arch, os_name)
        else:
            self.name = '%s-%s-%s' % (arch, os_name, variant)
        self.triplet = '%s-glibc-%s' % (arch, os_name)
        if gcc_cfg is None:
            self.gcc_cfg = []
        else:
            self.gcc_cfg = gcc_cfg
        if first_gcc_cfg is None:
            self.first_gcc_cfg = []
        else:
            self.first_gcc_cfg = first_gcc_cfg
        if binutils_cfg is None:
            self.binutils_cfg = []
        else:
            self.binutils_cfg = binutils_cfg
        if glibcs is None:
            glibcs = [{'variant': variant}]
        if extra_glibcs is None:
            extra_glibcs = []
        glibcs = [Glibc(self, **g) for g in glibcs]
        extra_glibcs = [Glibc(self, **g) for g in extra_glibcs]
        self.all_glibcs = glibcs + extra_glibcs
        self.compiler_glibcs = glibcs
        self.installdir = ctx.compiler_installdir(self.name)
        self.bindir = ctx.compiler_bindir(self.name)
        self.sysroot = ctx.compiler_sysroot(self.name)
        self.builddir = os.path.join(ctx.builddir, 'compilers', self.name)
        self.logsdir = os.path.join(ctx.logsdir, 'compilers', self.name)

    def component_builddir(self, component):
        """Return the directory to use for a (non-glibc) build."""
        return self.ctx.component_builddir('compilers', self.name, component)

    def build(self):
        """Generate commands to build this compiler."""
        self.ctx.remove_recreate_dirs(self.installdir, self.builddir,
                                      self.logsdir)
        cmdlist = CommandList('compilers-%s' % self.name, self.ctx.keep)
        cmdlist.add_command('check-host-libraries',
                            ['test', '-f',
                             os.path.join(self.ctx.host_libraries_installdir,
                                          'ok')])
        cmdlist.use_path(self.bindir)
        self.build_cross_tool(cmdlist, 'binutils', 'binutils',
                              ['--disable-gdb',
                               '--disable-gdbserver',
                               '--disable-libdecnumber',
                               '--disable-readline',
                               '--disable-sim'] + self.binutils_cfg)
        if self.os.startswith('linux'):
            install_linux_headers(LinuxHeadersPolicyForBuild(self), cmdlist)
        self.build_gcc(cmdlist, True)
        if self.os == 'gnu':
            self.install_gnumach_headers(cmdlist)
            self.build_cross_tool(cmdlist, 'mig', 'mig')
            self.install_hurd_headers(cmdlist)
        for g in self.compiler_glibcs:
            cmdlist.push_subdesc('glibc')
            cmdlist.push_subdesc(g.name)
            g.build_glibc(cmdlist, GlibcPolicyForCompiler(g))
            cmdlist.pop_subdesc()
            cmdlist.pop_subdesc()
        self.build_gcc(cmdlist, False)
        cmdlist.add_command('done', ['touch',
                                     os.path.join(self.installdir, 'ok')])
        self.ctx.add_makefile_cmdlist('compilers-%s' % self.name, cmdlist,
                                      self.logsdir)

    def build_cross_tool(self, cmdlist, tool_src, tool_build, extra_opts=None):
        """Build one cross tool."""
        srcdir = self.ctx.component_srcdir(tool_src)
        builddir = self.component_builddir(tool_build)
        cmdlist.push_subdesc(tool_build)
        cmdlist.create_use_dir(builddir)
        cfg_cmd = [os.path.join(srcdir, 'configure'),
                   '--prefix=%s' % self.installdir,
                   '--build=%s' % self.ctx.build_triplet,
                   '--host=%s' % self.ctx.build_triplet,
                   '--target=%s' % self.triplet,
                   '--with-sysroot=%s' % self.sysroot]
        if extra_opts:
            cfg_cmd.extend(extra_opts)
        cmdlist.add_command('configure', cfg_cmd)
        cmdlist.add_command('build', ['make'])
        # Parallel "make install" for GCC has race conditions that can
        # cause it to fail; see
        # <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=42980>.  Such
        # problems are not known for binutils, but doing the
        # installation in parallel within a particular toolchain build
        # (as opposed to installation of one toolchain from
        # build-many-glibcs.py running in parallel to the installation
        # of other toolchains being built) is not known to be
        # significantly beneficial, so it is simplest just to disable
        # parallel install for cross tools here.
        cmdlist.add_command('install', ['make', '-j1', 'install'])
        cmdlist.cleanup_dir()
        cmdlist.pop_subdesc()

    def install_gnumach_headers(self, cmdlist):
        """Install GNU Mach headers."""
        srcdir = self.ctx.component_srcdir('gnumach')
        builddir = self.component_builddir('gnumach')
        cmdlist.push_subdesc('gnumach')
        cmdlist.create_use_dir(builddir)
        cmdlist.add_command('configure',
                            [os.path.join(srcdir, 'configure'),
                             '--build=%s' % self.ctx.build_triplet,
                             '--host=%s' % self.triplet,
                             '--prefix=',
                             'CC=%s-gcc -nostdlib' % self.triplet])
        cmdlist.add_command('install', ['make', 'DESTDIR=%s' % self.sysroot,
                                        'install-data'])
        cmdlist.cleanup_dir()
        cmdlist.pop_subdesc()

    def install_hurd_headers(self, cmdlist):
        """Install Hurd headers."""
        srcdir = self.ctx.component_srcdir('hurd')
        builddir = self.component_builddir('hurd')
        cmdlist.push_subdesc('hurd')
        cmdlist.create_use_dir(builddir)
        cmdlist.add_command('configure',
                            [os.path.join(srcdir, 'configure'),
                             '--build=%s' % self.ctx.build_triplet,
                             '--host=%s' % self.triplet,
                             '--prefix=',
                             '--disable-profile', '--without-parted',
                             'CC=%s-gcc -nostdlib' % self.triplet])
        cmdlist.add_command('install', ['make', 'prefix=%s' % self.sysroot,
                                        'no_deps=t', 'install-headers'])
        cmdlist.cleanup_dir()
        cmdlist.pop_subdesc()

    def build_gcc(self, cmdlist, bootstrap):
        """Build GCC."""
        # libssp is of little relevance with glibc's own stack
        # checking support.  libcilkrts does not support GNU/Hurd (and
        # has been removed in GCC 8, so --disable-libcilkrts can be
        # removed once glibc no longer supports building with older
        # GCC versions).
        cfg_opts = list(self.gcc_cfg)
        cfg_opts += ['--disable-libssp', '--disable-libcilkrts']
        host_libs = self.ctx.host_libraries_installdir
        cfg_opts += ['--with-gmp=%s' % host_libs,
                     '--with-mpfr=%s' % host_libs,
                     '--with-mpc=%s' % host_libs]
        if bootstrap:
            tool_build = 'gcc-first'
            # Building a static-only, C-only compiler that is
            # sufficient to build glibc.  Various libraries and
            # features that may require libc headers must be disabled.
            # When configuring with a sysroot, --with-newlib is
            # required to define inhibit_libc (to stop some parts of
            # libgcc including libc headers); --without-headers is not
            # sufficient.
            cfg_opts += ['--enable-languages=c', '--disable-shared',
                         '--disable-threads',
                         '--disable-libatomic',
                         '--disable-decimal-float',
                         '--disable-libffi',
                         '--disable-libgomp',
                         '--disable-libitm',
                         '--disable-libmpx',
                         '--disable-libquadmath',
                         '--disable-libsanitizer',
                         '--without-headers', '--with-newlib',
                         '--with-glibc-version=%s' % self.ctx.glibc_version
                         ]
            cfg_opts += self.first_gcc_cfg
        else:
            tool_build = 'gcc'
            # libsanitizer commonly breaks because of glibc header
            # changes, or on unusual targets.  C++ pre-compiled
            # headers are not used during the glibc build and are
            # expensive to create.
            if not self.ctx.full_gcc:
                cfg_opts += ['--disable-libsanitizer',
                             '--disable-libstdcxx-pch']
            langs = 'all' if self.ctx.full_gcc else 'c,c++'
            cfg_opts += ['--enable-languages=%s' % langs,
                         '--enable-shared', '--enable-threads']
        self.build_cross_tool(cmdlist, 'gcc', tool_build, cfg_opts)

class GlibcPolicyDefault(object):
    """Build policy for glibc: common defaults."""

    def __init__(self, glibc):
        self.srcdir = glibc.ctx.component_srcdir('glibc')
        self.use_usr = glibc.os != 'gnu'
        self.prefix = '/usr' if self.use_usr else ''
        self.configure_args = [
            '--prefix=%s' % self.prefix,
            '--enable-profile',
            '--build=%s' % glibc.ctx.build_triplet,
            '--host=%s' % glibc.triplet,
            'CC=%s' % glibc.tool_name('gcc'),
            'CXX=%s' % glibc.tool_name('g++'),
            'AR=%s' % glibc.tool_name('ar'),
            'AS=%s' % glibc.tool_name('as'),
            'LD=%s' % glibc.tool_name('ld'),
            'NM=%s' % glibc.tool_name('nm'),
            'OBJCOPY=%s' % glibc.tool_name('objcopy'),
            'OBJDUMP=%s' % glibc.tool_name('objdump'),
            'RANLIB=%s' % glibc.tool_name('ranlib'),
            'READELF=%s' % glibc.tool_name('readelf'),
            'STRIP=%s' % glibc.tool_name('strip'),
        ]
        if glibc.os == 'gnu':
            self.configure_args.append('MIG=%s' % glibc.tool_name('mig'))
        if glibc.cflags:
            self.configure_args.append('CFLAGS=%s' % glibc.cflags)
            self.configure_args.append('CXXFLAGS=%s' % glibc.cflags)
        self.configure_args += glibc.cfg

    def configure(self, cmdlist):
        """Invoked to add the configure command to the command list."""
        cmdlist.add_command('configure',
                            [os.path.join(self.srcdir, 'configure'),
                             *self.configure_args])

    def extra_commands(self, cmdlist):
        """Invoked to inject additional commands (make check) after build."""
        pass

class GlibcPolicyForCompiler(GlibcPolicyDefault):
    """Build policy for glibc during the compilers stage."""

    def __init__(self, glibc):
        super().__init__(glibc)
        self.builddir = glibc.ctx.component_builddir(
            'compilers', glibc.compiler.name, 'glibc', glibc.name)
        self.installdir = glibc.compiler.sysroot

class GlibcPolicyForBuild(GlibcPolicyDefault):
    """Build policy for glibc during the glibcs stage."""

    def __init__(self, glibc):
        super().__init__(glibc)
        self.builddir = glibc.ctx.component_builddir(
            'glibcs', glibc.name, 'glibc')
        self.installdir = glibc.ctx.glibc_installdir(glibc.name)
        if glibc.ctx.strip:
            self.strip = glibc.tool_name('strip')
        else:
            self.strip = None
        self.save_logs = glibc.ctx.save_logs

    def extra_commands(self, cmdlist):
        if self.strip:
            # Avoid picking up libc.so and libpthread.so, which are
            # linker scripts stored in /lib on Hurd.  libc and
            # libpthread are still stripped via their libc-X.YY.so
            # implementation files.
            find_command = (('find %s/lib* -name "*.so"'
                             + r' \! -name libc.so \! -name libpthread.so')
                            % self.installdir)
            cmdlist.add_command('strip', ['sh', '-c', ('%s $(%s)' %
                                  (self.strip, find_command))])
        cmdlist.add_command('check', ['make', 'check'])
        cmdlist.add_command('save-logs', [self.save_logs], always_run=True)

class GlibcPolicyForUpdateSyscalls(GlibcPolicyDefault):
    """Build policy for glibc during update-syscalls."""

    def __init__(self, glibc):
        super().__init__(glibc)
        self.builddir = glibc.ctx.component_builddir(
            'update-syscalls', glibc.name, 'glibc')
        self.linuxdir = glibc.ctx.component_builddir(
            'update-syscalls', glibc.name, 'linux')
        self.linux_policy = LinuxHeadersPolicyForUpdateSyscalls(
            glibc, self.linuxdir)
        self.configure_args.insert(
            0, '--with-headers=%s' % os.path.join(self.linuxdir, 'include'))
        # self.installdir not set because installation is not supported

class Glibc(object):
    """A configuration for building glibc."""

    def __init__(self, compiler, arch=None, os_name=None, variant=None,
                 cfg=None, ccopts=None, cflags=None):
        """Initialize a Glibc object."""
        self.ctx = compiler.ctx
        self.compiler = compiler
        if arch is None:
            self.arch = compiler.arch
        else:
            self.arch = arch
        if os_name is None:
            self.os = compiler.os
        else:
            self.os = os_name
        self.variant = variant
        if variant is None:
            self.name = '%s-%s' % (self.arch, self.os)
        else:
            self.name = '%s-%s-%s' % (self.arch, self.os, variant)
        self.triplet = '%s-glibc-%s' % (self.arch, self.os)
        if cfg is None:
            self.cfg = []
        else:
            self.cfg = cfg
        # ccopts contain ABI options and are passed to configure as CC / CXX.
        self.ccopts = ccopts
        # cflags contain non-ABI options like -g or -O and are passed to
        # configure as CFLAGS / CXXFLAGS.
        self.cflags = cflags

    def tool_name(self, tool):
        """Return the name of a cross-compilation tool."""
        ctool = '%s-%s' % (self.compiler.triplet, tool)
        if self.ccopts and (tool == 'gcc' or tool == 'g++'):
            ctool = '%s %s' % (ctool, self.ccopts)
        return ctool

    def build(self):
        """Generate commands to build this glibc."""
        builddir = self.ctx.component_builddir('glibcs', self.name, 'glibc')
        installdir = self.ctx.glibc_installdir(self.name)
        logsdir = os.path.join(self.ctx.logsdir, 'glibcs', self.name)
        self.ctx.remove_recreate_dirs(installdir, builddir, logsdir)
        cmdlist = CommandList('glibcs-%s' % self.name, self.ctx.keep)
        cmdlist.add_command('check-compilers',
                            ['test', '-f',
                             os.path.join(self.compiler.installdir, 'ok')])
        cmdlist.use_path(self.compiler.bindir)
        self.build_glibc(cmdlist, GlibcPolicyForBuild(self))
        self.ctx.add_makefile_cmdlist('glibcs-%s' % self.name, cmdlist,
                                      logsdir)

    def build_glibc(self, cmdlist, policy):
        """Generate commands to build this glibc, either as part of a compiler
        build or with the bootstrapped compiler (and in the latter case, run
        tests as well)."""
        cmdlist.create_use_dir(policy.builddir)
        policy.configure(cmdlist)
        cmdlist.add_command('build', ['make'])
        cmdlist.add_command('install', ['make', 'install',
                                        'install_root=%s' % policy.installdir])
        # GCC uses paths such as lib/../lib64, so make sure lib
        # directories always exist.
        mkdir_cmd = ['mkdir', '-p',
                     os.path.join(policy.installdir, 'lib')]
        if policy.use_usr:
            mkdir_cmd += [os.path.join(policy.installdir, 'usr', 'lib')]
        cmdlist.add_command('mkdir-lib', mkdir_cmd)
        policy.extra_commands(cmdlist)
        cmdlist.cleanup_dir()

    def update_syscalls(self):
        if self.os == 'gnu':
            # Hurd does not have system call tables that need updating.
            return

        policy = GlibcPolicyForUpdateSyscalls(self)
        logsdir = os.path.join(self.ctx.logsdir, 'update-syscalls', self.name)
        self.ctx.remove_recreate_dirs(policy.builddir, logsdir)
        cmdlist = CommandList('update-syscalls-%s' % self.name, self.ctx.keep)
        cmdlist.add_command('check-compilers',
                            ['test', '-f',
                             os.path.join(self.compiler.installdir, 'ok')])
        cmdlist.use_path(self.compiler.bindir)

        install_linux_headers(policy.linux_policy, cmdlist)

        cmdlist.create_use_dir(policy.builddir)
        policy.configure(cmdlist)
        cmdlist.add_command('build', ['make', 'update-syscall-lists'])
        cmdlist.cleanup_dir()
        self.ctx.add_makefile_cmdlist('update-syscalls-%s' % self.name,
                                      cmdlist, logsdir)

class Command(object):
    """A command run in the build process."""

    def __init__(self, desc, num, dir, path, command, always_run=False):
        """Initialize a Command object."""
        self.dir = dir
        self.path = path
        self.desc = desc
        trans = str.maketrans({' ': '-'})
        self.logbase = '%03d-%s' % (num, desc.translate(trans))
        self.command = command
        self.always_run = always_run

    @staticmethod
    def shell_make_quote_string(s):
        """Given a string not containing a newline, quote it for use by the
        shell and make."""
        assert '\n' not in s
        if re.fullmatch('[]+,./0-9@A-Z_a-z-]+', s):
            return s
        strans = str.maketrans({"'": "'\\''"})
        s = "'%s'" % s.translate(strans)
        mtrans = str.maketrans({'$': '$$'})
        return s.translate(mtrans)

    @staticmethod
    def shell_make_quote_list(l, translate_make):
        """Given a list of strings not containing newlines, quote them for use
        by the shell and make, returning a single string.  If translate_make
        is true and the first string is 'make', change it to $(MAKE)."""
        l = [Command.shell_make_quote_string(s) for s in l]
        if translate_make and l[0] == 'make':
            l[0] = '$(MAKE)'
        return ' '.join(l)

    def shell_make_quote(self):
        """Return this command quoted for the shell and make."""
        return self.shell_make_quote_list(self.command, True)


class CommandList(object):
    """A list of commands run in the build process."""

    def __init__(self, desc, keep):
        """Initialize a CommandList object."""
        self.cmdlist = []
        self.dir = None
        self.path = None
        self.desc = [desc]
        self.keep = keep

    def desc_txt(self, desc):
        """Return the description to use for a command."""
        return '%s %s' % (' '.join(self.desc), desc)

    def use_dir(self, dir):
        """Set the default directory for subsequent commands."""
        self.dir = dir

    def use_path(self, path):
        """Set a directory to be prepended to the PATH for subsequent
        commands."""
        self.path = path

    def push_subdesc(self, subdesc):
        """Set the default subdescription for subsequent commands (e.g., the
        name of a component being built, within the series of commands
        building it)."""
        self.desc.append(subdesc)

    def pop_subdesc(self):
        """Pop a subdescription from the list of descriptions."""
        self.desc.pop()

    def create_use_dir(self, dir):
        """Remove and recreate a directory and use it for subsequent
        commands."""
        self.add_command_dir('rm', None, ['rm', '-rf', dir])
        self.add_command_dir('mkdir', None, ['mkdir', '-p', dir])
        self.use_dir(dir)

    def add_command_dir(self, desc, dir, command, always_run=False):
        """Add a command to run in a given directory."""
        cmd = Command(self.desc_txt(desc), len(self.cmdlist), dir, self.path,
                      command, always_run)
        self.cmdlist.append(cmd)

    def add_command(self, desc, command, always_run=False):
        """Add a command to run in the default directory."""
        cmd = Command(self.desc_txt(desc), len(self.cmdlist), self.dir,
                      self.path, command, always_run)
        self.cmdlist.append(cmd)

    def cleanup_dir(self, desc='cleanup', dir=None):
        """Clean up a build directory.  If no directory is specified, the
        default directory is cleaned up and ceases to be the default
        directory."""
        if dir is None:
            dir = self.dir
            self.use_dir(None)
        if self.keep != 'all':
            self.add_command_dir(desc, None, ['rm', '-rf', dir],
                                 always_run=(self.keep == 'none'))

    def makefile_commands(self, wrapper, logsdir):
        """Return the sequence of commands in the form of text for a Makefile.
        The given wrapper script takes arguments: base of logs for
        previous command, or empty; base of logs for this command;
        description; directory; PATH addition; the command itself."""
        # prev_base is the base of the name for logs of the previous
        # command that is not always-run (that is, a build command,
        # whose failure should stop subsequent build commands from
        # being run, as opposed to a cleanup command, which is run
        # even if previous commands failed).
        prev_base = ''
        cmds = []
        for c in self.cmdlist:
            ctxt = c.shell_make_quote()
            if prev_base and not c.always_run:
                prev_log = os.path.join(logsdir, prev_base)
            else:
                prev_log = ''
            this_log = os.path.join(logsdir, c.logbase)
            if not c.always_run:
                prev_base = c.logbase
            if c.dir is None:
                dir = ''
            else:
                dir = c.dir
            if c.path is None:
                path = ''
            else:
                path = c.path
            prelims = [wrapper, prev_log, this_log, c.desc, dir, path]
            prelim_txt = Command.shell_make_quote_list(prelims, False)
            cmds.append('\t@%s %s' % (prelim_txt, ctxt))
        return '\n'.join(cmds)

    def status_logs(self, logsdir):
        """Return the list of log files with command status."""
        return [os.path.join(logsdir, '%s-status.txt' % c.logbase)
                for c in self.cmdlist]


def get_parser():
    """Return an argument parser for this module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-j', dest='parallelism',
                        help='Run this number of jobs in parallel',
                        type=int, default=os.cpu_count())
    parser.add_argument('--keep', dest='keep',
                        help='Whether to keep all build directories, '
                        'none or only those from failed builds',
                        default='none', choices=('none', 'all', 'failed'))
    parser.add_argument('--replace-sources', action='store_true',
                        help='Remove and replace source directories '
                        'with the wrong version of a component')
    parser.add_argument('--strip', action='store_true',
                        help='Strip installed glibc libraries')
    parser.add_argument('--full-gcc', action='store_true',
                        help='Build GCC with all languages and libsanitizer')
    parser.add_argument('--shallow', action='store_true',
                        help='Do not download Git history during checkout')
    parser.add_argument('topdir',
                        help='Toplevel working directory')
    parser.add_argument('action',
                        help='What to do',
                        choices=('checkout', 'bot-cycle', 'bot',
                                 'host-libraries', 'compilers', 'glibcs',
                                 'update-syscalls', 'list-compilers',
                                 'list-glibcs'))
    parser.add_argument('configs',
                        help='Versions to check out or configurations to build',
                        nargs='*')
    return parser


def main(argv):
    """The main entry point."""
    parser = get_parser()
    opts = parser.parse_args(argv)
    topdir = os.path.abspath(opts.topdir)
    ctx = Context(topdir, opts.parallelism, opts.keep, opts.replace_sources,
                  opts.strip, opts.full_gcc, opts.action,
                  shallow=opts.shallow)
    ctx.run_builds(opts.action, opts.configs)


if __name__ == '__main__':
    main(sys.argv[1:])
