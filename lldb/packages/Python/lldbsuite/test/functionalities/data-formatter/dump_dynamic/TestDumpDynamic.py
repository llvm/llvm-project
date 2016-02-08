from __future__ import absolute_import

from lldbsuite.test import lldbinline

lldbinline.MakeInlineTest(__file__, globals(), [lldbinline.expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24663")])
