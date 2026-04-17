# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Entry point for LLDB to register data formatters.

import lldb
from libcxx_map_formatter import *
from libcxx_vector_formatter import *


def register_synthetic(
    debugger: lldb.SBDebugger, regex: str, class_name: str, extra_flags: str = ""
):
    debugger.HandleCommand(
        f'type synthetic add {extra_flags} -x "{regex}" -l {__name__}.{class_name} -w "cplusplus-py"'
    )


# Entry point for LLDB when importing scripts.
def __lldb_init_module(debugger, dict):
    register_synthetic(
        debugger, "^std::__[[:alnum:]]+::map<.+> >$", "LibcxxStdMapSyntheticProvider"
    )
    register_synthetic(
        debugger, "^std::__[[:alnum:]]+::set<.+> >$", "LibcxxStdMapSyntheticProvider"
    )
    register_synthetic(
        debugger,
        "^std::__[[:alnum:]]+::multiset<.+> >$",
        "LibcxxStdMapSyntheticProvider",
    )
    register_synthetic(
        debugger,
        "^std::__[[:alnum:]]+::multimap<.+> >$",
        "LibcxxStdMapSyntheticProvider",
    )
    register_synthetic(
        debugger,
        "^std::__[[:alnum:]]+::__map_(const_)?iterator<.+>$",
        "LibCxxMapIteratorSyntheticProvider",
    )
    register_synthetic(
        debugger,
        "^std::__[[:alnum:]]+::vector<.+>$",
        "LibCxxStdVectorSyntheticFrontendCreator",
        "--wants-dereference",
    )

    # Enables registered formatters in LLDB. The builtin LLDB formatters are registered in the `cplusplus` category.
    # We don't want to override/clash with those.
    debugger.HandleCommand("type category enable cplusplus-py")
