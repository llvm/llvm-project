from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test.decorators import *
from lldbsuite.test import configuration


@skipUnlessArm64eSupported
class Arm64eTestBase(TestBase):
    def build(self):
        super().build(
            dictionary={"TRIPLE": configuration.triple.replace("arm64-", "arm64e-")}
        )
