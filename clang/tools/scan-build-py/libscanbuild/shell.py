# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
""" This module implements basic shell escaping/unescaping methods. """

import re
import shlex

__all__ = ['encode', 'decode']


def encode(command):
    """ Takes a command as list and returns a string. """

    def needs_quote(word):
        """ Returns true if arguments needs to be protected by quotes.

        Previous implementation was shlex.split method, but that's not good
        for this job. Currently is running through the string with a basic
        state checking. """

        reserved = {' ', '$', '%', '&', '(', ')', '[', ']', '{', '}', '*', '|',
                    '<', '>', '@', '?', '!'}
        state = 0
        for current in word:
            if state == 0 and current in reserved:
                return True
            elif state == 0 and current == '\\':
                state = 1
            elif state == 1 and current in reserved | {'\\'}:
                state = 0
            elif state == 0 and current == '"':
                state = 2
            elif state == 2 and current == '"':
                state = 0
            elif state == 0 and current == "'":
                state = 3
            elif state == 3 and current == "'":
                state = 0
        return state != 0

    def escape(word):
        """ Do protect argument if that's needed. """

        table = {'\\': '\\\\', '"': '\\"'}
        escaped = ''.join([table.get(c, c) for c in word])

        return '"' + escaped + '"' if needs_quote(word) else escaped

    return " ".join([escape(arg) for arg in command])


def decode(string):
    """ Takes a command string and returns as a list. """

    def unescape(arg):
        """ Gets rid of the escaping characters. """

        if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] == '"':
            arg = arg[1:-1]
            return re.sub(r'\\(["\\])', r'\1', arg)
        return re.sub(r'\\([\\ $%&\(\)\[\]\{\}\*|<>@?!])', r'\1', arg)

    return [unescape(arg) for arg in shlex.split(string)]
