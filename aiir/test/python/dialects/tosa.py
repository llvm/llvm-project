# RUN: %PYTHON %s

from aiir.ir import *
import aiir.dialects.tosa as tosa


# Just make sure the dialect is populated with generated ops.
assert tosa.AddOp
