# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Provides Windows implementation of formatted/colored console output."""

import sys

import ctypes
import ctypes.wintypes

from ..PrettyOutputBase import PrettyOutputBase, Stream, _lock, _null_lock


class _CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
    # pylint: disable=protected-access
    _fields_ = [
        ("dwSize", ctypes.wintypes._COORD),
        ("dwCursorPosition", ctypes.wintypes._COORD),
        ("wAttributes", ctypes.c_ushort),
        ("srWindow", ctypes.wintypes._SMALL_RECT),
        ("dwMaximumWindowSize", ctypes.wintypes._COORD),
    ]
    # pylint: enable=protected-access


class PrettyOutput(PrettyOutputBase):

    stdout = Stream(sys.stdout, ctypes.windll.kernel32.GetStdHandle(-11))
    stderr = Stream(sys.stderr, ctypes.windll.kernel32.GetStdHandle(-12))

    def __enter__(self):
        info = _CONSOLE_SCREEN_BUFFER_INFO()

        for s in (PrettyOutput.stdout, PrettyOutput.stderr):
            ctypes.windll.kernel32.GetConsoleScreenBufferInfo(s.os, ctypes.byref(info))
            s.orig_color = info.wAttributes

        return self

    def __exit__(self, *args):
        self._restore_orig_color(PrettyOutput.stdout)
        self._restore_orig_color(PrettyOutput.stderr)

    def _restore_orig_color(self, stream, lock=_lock):
        if not stream.color_enabled:
            return

        with lock:
            stream = self._set_valid_stream(stream)
            self.flush(stream)
            if stream.orig_color:
                ctypes.windll.kernel32.SetConsoleTextAttribute(
                    stream.os, stream.orig_color
                )

    def _color(self, text, color, stream, lock=_lock):
        stream = self._set_valid_stream(stream)
        with lock:
            try:
                if stream.color_enabled:
                    ctypes.windll.kernel32.SetConsoleTextAttribute(stream.os, color)
                self._write(text, stream)
            finally:
                if stream.color_enabled:
                    self._restore_orig_color(stream, lock=_null_lock)

    def red_impl(self, text, stream=None, **kwargs):
        self._color(text, 12, stream, **kwargs)

    def yellow_impl(self, text, stream=None, **kwargs):
        self._color(text, 14, stream, **kwargs)

    def green_impl(self, text, stream=None, **kwargs):
        self._color(text, 10, stream, **kwargs)

    def blue_impl(self, text, stream=None, **kwargs):
        self._color(text, 11, stream, **kwargs)

    def default_impl(self, text, stream=None, **kwargs):
        stream = self._set_valid_stream(stream)
        self._color(text, stream.orig_color, stream, **kwargs)
