# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
"""
This module responsible to run the Clang static analyzer against any build
and generate reports.
"""


def duplicate_check(method):
    """ Predicate to detect duplicated entries.

    Unique hash method can be use to detect duplicates. Entries are
    represented as dictionaries, which has no default hash method.
    This implementation uses a set datatype to store the unique hash values.

    This method returns a method which can detect the duplicate values. """

    def predicate(entry):
        entry_hash = predicate.unique(entry)
        if entry_hash not in predicate.state:
            predicate.state.add(entry_hash)
            return False
        return True

    predicate.unique = method
    predicate.state = set()
    return predicate


def tempdir():
    """ Return the default temorary directory. """

    from os import getenv
    return getenv('TMPDIR', getenv('TEMP', getenv('TMP', '/tmp')))


def initialize_logging(verbose_level):
    """ Output content controlled by the verbosity level. """

    import sys
    import os.path
    import logging
    level = logging.WARNING - min(logging.WARNING, (10 * verbose_level))

    if verbose_level <= 3:
        fmt_string = '{0}: %(levelname)s: %(message)s'
    else:
        fmt_string = '{0}: %(levelname)s: %(funcName)s: %(message)s'

    program = os.path.basename(sys.argv[0])
    logging.basicConfig(format=fmt_string.format(program), level=level)


def command_entry_point(function):
    """ Decorator for command entry points. """

    import functools
    import logging

    @functools.wraps(function)
    def wrapper(*args, **kwargs):

        exit_code = 127
        try:
            exit_code = function(*args, **kwargs)
        except KeyboardInterrupt:
            logging.warning('Keyboard interupt')
        except Exception:
            logging.exception('Internal error.')
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.error("Please report this bug and attach the output "
                              "to the bug report")
            else:
                logging.error("Please run this command again and turn on "
                              "verbose mode (add '-vvv' as argument).")
        finally:
            return exit_code

    return wrapper
