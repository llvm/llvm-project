""" This module represents an abstraction of an lldb target / host platform. """

from __future__ import absolute_import

# System modules
import itertools

# Third-party modules
import six

# LLDB modules
import use_lldb_suite
import lldb

windows, linux, macosx, darwin, ios, darwin_all, freebsd, netbsd, bsd_all, android = range(
    10)

__name_lookup = {
    windows: ["windows"],
    linux: ["linux"],
    macosx: ["macosx"],
    darwin: ["darwin"],
    ios: ["ios"],
    darwin_all: ["macosx", "darwin", "ios"],
    freebsd: ["freebsd"],
    netbsd: ["netbsd"],
    bsd_all: ["freebsd", "netbsd"],
    android: ["android"]
}


def translate(values):

    if isinstance(values, six.integer_types):
        # This is a value from the platform enumeration, translate it.
        return __name_lookup[values]
    elif isinstance(values, six.string_types):
        # This is a raw string, return it.
        return [values]
    elif hasattr(values, "__iter__"):
        # This is an iterable, convert each item.
        result = [translate(x) for x in values]
        result = list(itertools.chain(*result))
        return result
    return values
