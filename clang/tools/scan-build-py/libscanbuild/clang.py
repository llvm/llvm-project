# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
""" This module is responsible for the Clang executable.

Since Clang command line interface is so rich, but this project is using only
a subset of that, it makes sense to create a function specific wrapper. """

import re
from libscanbuild import run_command
from libscanbuild.shell import decode

__all__ = ['get_version', 'get_arguments', 'get_checkers']

# regex for activated checker
ACTIVE_CHECKER_PATTERN = re.compile(r'^-analyzer-checker=(.*)$')


def get_version(clang):
    """ Returns the compiler version as string.

    :param clang:   the compiler we are using
    :return:        the version string printed to stderr """

    output = run_command([clang, '-v'])
    # the relevant version info is in the first line
    return output[0]


def get_arguments(command, cwd):
    """ Capture Clang invocation.

    :param command: the compilation command
    :param cwd:     the current working directory
    :return:        the detailed front-end invocation command """

    cmd = command[:]
    cmd.insert(1, '-###')

    output = run_command(cmd, cwd=cwd)
    # The relevant information is in the last line of the output.
    # Don't check if finding last line fails, would throw exception anyway.
    last_line = output[-1]
    if re.search(r'clang(.*): error:', last_line):
        raise Exception(last_line)
    return decode(last_line)


def get_active_checkers(clang, plugins):
    """ Get the active checker list.

    :param clang:   the compiler we are using
    :param plugins: list of plugins which was requested by the user
    :return:        list of checker names which are active

    To get the default checkers we execute Clang to print how this
    compilation would be called. And take out the enabled checker from the
    arguments. For input file we specify stdin and pass only language
    information. """

    def get_active_checkers_for(language):
        """ Returns a list of active checkers for the given language. """

        load_args = [arg
                     for plugin in plugins
                     for arg in ['-Xclang', '-load', '-Xclang', plugin]]
        cmd = [clang, '--analyze'] + load_args + ['-x', language, '-']
        return [ACTIVE_CHECKER_PATTERN.match(arg).group(1)
                for arg in get_arguments(cmd, '.')
                if ACTIVE_CHECKER_PATTERN.match(arg)]

    result = set()
    for language in ['c', 'c++', 'objective-c', 'objective-c++']:
        result.update(get_active_checkers_for(language))
    return frozenset(result)


def is_active(checkers):
    """ Returns a method, which classifies the checker active or not,
    based on the received checker name list. """

    def predicate(checker):
        """ Returns True if the given checker is active. """

        return any(pattern.match(checker) for pattern in predicate.patterns)

    predicate.patterns = [re.compile(r'^' + a + r'(\.|$)') for a in checkers]
    return predicate


def parse_checkers(stream):
    """ Parse clang -analyzer-checker-help output.

    Below the line 'CHECKERS:' are there the name description pairs.
    Many of them are in one line, but some long named checker has the
    name and the description in separate lines.

    The checker name is always prefixed with two space character. The
    name contains no whitespaces. Then followed by newline (if it's
    too long) or other space characters comes the description of the
    checker. The description ends with a newline character.

    :param stream:  list of lines to parse
    :return:        generator of tuples

    (<checker name>, <checker description>) """

    lines = iter(stream)
    # find checkers header
    for line in lines:
        if re.match(r'^CHECKERS:', line):
            break
    # find entries
    state = None
    for line in lines:
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


def get_checkers(clang, plugins):
    """ Get all the available checkers from default and from the plugins.

    :param clang:   the compiler we are using
    :param plugins: list of plugins which was requested by the user
    :return:        a dictionary of all available checkers and its status

    {<checker name>: (<checker description>, <is active by default>)} """

    load = [elem for plugin in plugins for elem in ['-load', plugin]]
    cmd = [clang, '-cc1'] + load + ['-analyzer-checker-help']

    lines = run_command(cmd)

    is_active_checker = is_active(get_active_checkers(clang, plugins))

    checkers = {
        name: (description, is_active_checker(name))
        for name, description in parse_checkers(lines)
    }
    if not checkers:
        raise Exception('Could not query Clang for available checkers.')

    return checkers
