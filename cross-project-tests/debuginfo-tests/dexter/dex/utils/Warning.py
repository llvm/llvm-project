# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utility functions for producing command line warnings."""


def warn(context, msg, flag=None):
    if context.options.no_warnings:
        return

    msg = msg.rstrip()
    if flag:
        msg = '{} <y>[{}]</>'.format(msg, flag)

    context.o.auto('warning: <d>{}</>\n'.format(msg))
