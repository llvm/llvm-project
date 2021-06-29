# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Base class for all DExTer commands, where a command is a specific Python
function that can be embedded into a comment in the source code under test
which will then be executed by DExTer during debugging.
"""

import abc
from typing import List

class CommandBase(object, metaclass=abc.ABCMeta):
    def __init__(self):
        self.path = None
        self.lineno = None
        self.raw_text = ''

    def get_label_args(self):
        return list()

    def has_labels(self):
        return False

    @abc.abstractstaticmethod
    def get_name():
        """This abstract method is usually implemented in subclasses as:
        return __class__.__name__
        """

    def get_watches(self) -> List[str]:
        return []

    @abc.abstractmethod
    def eval(self):
        """Evaluate the command.

        This will be called when constructing a Heuristic object to determine
        the debug score.

        Returns:
            The logic for handling the result of CommandBase.eval() must be
            defined in Heuristic.__init__() so a consitent return type between
            commands is not enforced.
        """

    @staticmethod
    def get_subcommands() -> dict:
        """Returns a dictionary of subcommands in the form {name: command} or
        None if no subcommands are required.
        """
        return None
