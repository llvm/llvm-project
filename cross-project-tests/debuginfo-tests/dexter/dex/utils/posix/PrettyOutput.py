# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Provides POSIX implementation of formatted/colored console output."""

from ..PrettyOutputBase import PrettyOutputBase, _lock


class PrettyOutput(PrettyOutputBase):
    def _color(self, text, color, stream, lock=_lock):
        """Use ANSI escape codes to provide color on Linux."""
        stream = self._set_valid_stream(stream)
        with lock:
            if stream.color_enabled:
                text = "\033[{}m{}\033[0m".format(color, text)
            self._write(text, stream)

    def red_impl(self, text, stream=None, **kwargs):
        self._color(text, 91, stream, **kwargs)

    def yellow_impl(self, text, stream=None, **kwargs):
        self._color(text, 93, stream, **kwargs)

    def green_impl(self, text, stream=None, **kwargs):
        self._color(text, 92, stream, **kwargs)

    def blue_impl(self, text, stream=None, **kwargs):
        self._color(text, 96, stream, **kwargs)

    def default_impl(self, text, stream=None, **kwargs):
        self._color(text, 0, stream, **kwargs)
