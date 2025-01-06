# ====- Information about standard headers used by docgen  ----*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#
from pathlib import Path
from typing import Generator


class Header:
    """
    Maintains implementation information about a standard header file:
    * where does its implementation dir live
    * where is its macros file
    * where is its docgen yaml file

    By convention, the macro-only part of a header file is in a header-specific
    file somewhere in the directory tree with root at
    ``$LLVM_PROJECT_ROOT/libc/include/llvm-libc-macros``.  Docgen expects that
    if a macro is implemented, that it appears in a string
    ``#define MACRO_NAME`` in some ``*-macros.h`` file in the directory tree.
    Docgen searches for this string in the file to set the implementation status
    shown in the generated rst docs rendered as html for display at
    <libc.llvm.org>.

    By convention, each function for a header is implemented in a function-specific
    cpp file somewhere in the directory tree with root at, e.g,
    ``$LLVM_PROJECT_ROOT/libc/src/fenv``. Some headers have architecture-specific
    implementations, like ``math``, and some don't, like ``fenv``. Docgen uses the
    presence of this function-specific cpp file to set the implementation status
    shown in the generated rst docs rendered as html for display at
    <libc.llvm.org>.
    """

    def __init__(self, header_name: str):
        """
        :param header_name: e.g., ``"threads.h"`` or ``"signal.h"``
        """
        self.name = header_name
        self.stem = header_name.rstrip(".h")
        self.docgen_root = Path(__file__).parent
        self.libc_root = self.docgen_root.parent.parent
        self.docgen_yaml = self.docgen_root / Path(header_name).with_suffix(".yaml")
        self.fns_dir = Path(self.libc_root, "src", self.stem)
        self.macros_dir = Path(self.libc_root, "include", "llvm-libc-macros")

    def macro_file_exists(self) -> bool:
        for _ in self.__get_macro_files():
            return True

        return False

    def fns_dir_exists(self) -> bool:
        return self.fns_dir.exists() and self.fns_dir.is_dir()

    def implements_fn(self, fn_name: str) -> bool:
        for _ in self.fns_dir.glob(f"**/{fn_name}.cpp"):
            return True

        return False

    def implements_macro(self, m_name: str) -> bool:
        """
        Some macro files are in, e.g.,
        ``$LLVM_PROJECT_ROOT/libc/include/llvm-libc-macros/fenv-macros.h``,
        but others are in subdirectories, e.g., ``signal.h`` has the macro
        definitions in
        ``$LLVM_PROJECT_ROOT/libc/include/llvm-libc-macros/linux/signal-macros.h``.

        :param m_name: name of macro, e.g., ``FE_ALL_EXCEPT``
        """
        for f in self.__get_macro_files():
            if f"#define {m_name}" in f.read_text():
                return True

        return False

    def __get_macro_files(self) -> Generator[Path, None, None]:
        """
        This function uses a glob on, e.g., ``"**/fcntl.macros.h"`` because the
        macro file might be located in a subdirectory:
        libc/include/llvm-libc-macros/fcntl-macros.h
        libc/include/llvm-libc-macros/linux/fcntl-macros.h

        When a header would be nested in a dir (such as arpa/, sys/, etc) we
        instead use a hyphen in the name.
        libc/include/llvm-libc-macros/sys-mman-macros.h
        """
        stem = self.stem.replace("/", "-")
        return self.macros_dir.glob(f"**/{stem}-macros.h")
