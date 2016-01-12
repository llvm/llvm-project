# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
""" This module is responsible for the Clang executable.

Since Clang command line interface is so rich, but this project is using only
a subset of that, it makes sense to create a function specific wrapper. """

import re
import subprocess
import logging
from libscanbuild.shell import decode

__all__ = ['get_version', 'get_arguments', 'get_checkers']


def get_version(cmd):
    """ Returns the compiler version as string. """

    lines = subprocess.check_output([cmd, '-v'], stderr=subprocess.STDOUT)
    return lines.decode('ascii').splitlines()[0]


def get_arguments(command, cwd):
    """ Capture Clang invocation.

    This method returns the front-end invocation that would be executed as
    a result of the given driver invocation. """

    def lastline(stream):
        last = None
        for line in stream:
            last = line
        if last is None:
            raise Exception("output not found")
        return last

    cmd = command[:]
    cmd.insert(1, '-###')
    logging.debug('exec command in %s: %s', cwd, ' '.join(cmd))
    child = subprocess.Popen(cmd,
                             cwd=cwd,
                             universal_newlines=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    line = lastline(child.stdout)
    child.stdout.close()
    child.wait()
    if child.returncode == 0:
        if re.search(r'clang(.*): error:', line):
            raise Exception(line)
        return decode(line)
    else:
        raise Exception(line)


def get_active_checkers(clang, plugins):
    """ To get the default plugins we execute Clang to print how this
    compilation would be called.

    For input file we specify stdin and pass only language information. """

    def checkers(language):
        """ Returns a list of active checkers for the given language. """

        load = [elem
                for plugin in plugins
                for elem in ['-Xclang', '-load', '-Xclang', plugin]]
        cmd = [clang, '--analyze'] + load + ['-x', language, '-']
        pattern = re.compile(r'^-analyzer-checker=(.*)$')
        return [pattern.match(arg).group(1)
                for arg in get_arguments(cmd, '.') if pattern.match(arg)]

    result = set()
    for language in ['c', 'c++', 'objective-c', 'objective-c++']:
        result.update(checkers(language))
    return result


def get_checkers(clang, plugins):
    """ Get all the available checkers from default and from the plugins.

    clang -- the compiler we are using
    plugins -- list of plugins which was requested by the user

    This method returns a dictionary of all available checkers and status.

    {<plugin name>: (<plugin description>, <is active by default>)} """

    plugins = plugins if plugins else []

    def parse_checkers(stream):
        """ Parse clang -analyzer-checker-help output.

        Below the line 'CHECKERS:' are there the name description pairs.
        Many of them are in one line, but some long named plugins has the
        name and the description in separate lines.

        The plugin name is always prefixed with two space character. The
        name contains no whitespaces. Then followed by newline (if it's
        too long) or other space characters comes the description of the
        plugin. The description ends with a newline character. """

        # find checkers header
        for line in stream:
            if re.match(r'^CHECKERS:', line):
                break
        # find entries
        state = None
        for line in stream:
            if state and not re.match(r'^\s\s\S', line):
                yield (state, line.strip())
                state = None
            elif re.match(r'^\s\s\S+$', line.rstrip()):
                state = line.strip()
            else:
                pattern = re.compile(r'^\s\s(?P<key>\S*)\s*(?P<value>.*)')
                match = pattern.match(line.rstrip())
                if match:
                    current = match.groupdict()
                    yield (current['key'], current['value'])

    def is_active(actives, entry):
        """ Returns true if plugin name is matching the active plugin names.

        actives -- set of active plugin names (or prefixes).
        entry -- the current plugin name to judge.

        The active plugin names are specific plugin names or prefix of some
        names. One example for prefix, when it say 'unix' and it shall match
        on 'unix.API', 'unix.Malloc' and 'unix.MallocSizeof'. """

        return any(re.match(r'^' + a + r'(\.|$)', entry) for a in actives)

    actives = get_active_checkers(clang, plugins)

    load = [elem for plugin in plugins for elem in ['-load', plugin]]
    cmd = [clang, '-cc1'] + load + ['-analyzer-checker-help']

    logging.debug('exec command: %s', ' '.join(cmd))
    child = subprocess.Popen(cmd,
                             universal_newlines=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    checkers = {
        k: (v, is_active(actives, k))
        for k, v in parse_checkers(child.stdout)
    }
    child.stdout.close()
    child.wait()
    if child.returncode == 0 and len(checkers):
        return checkers
    else:
        raise Exception('Could not query Clang for available checkers.')
