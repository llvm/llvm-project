from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(__file__,
                          globals(),
                          [decorators.expectedFailureAll(compiler="clang",
                                                         compiler_version=["<",
                                                                           "3.5"],
                                                         bugnumber="llvm.org/pr27845"),
                           decorators.expectedFailureAll(archs="aarch64",
                                                         oslist="linux",
                                                         debug_info="dwo",
                                                         bugnumber="llvm.org/pr44037")])
