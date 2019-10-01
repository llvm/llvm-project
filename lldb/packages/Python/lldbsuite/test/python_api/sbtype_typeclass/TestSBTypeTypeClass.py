from lldbsuite.test import decorators
from lldbsuite.test import lldbinline

lldbinline.MakeInlineTest(
    __file__, globals(), [
        decorators.skipIfFreeBSD, decorators.skipIfLinux,
        decorators.skipIfWindows, decorators.skipIfNetBSD,
        decorators.expectedFailureAll(
            oslist=['macosx'], archs=['i386'],
            bugnumber='rdar://28656677')])
