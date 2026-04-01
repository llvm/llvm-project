# RUN: %PYTHON %s | FileCheck %s

from aiir.ir import *
from aiir.dialects import arith
import aiir.dialects.async_dialect as async_dialect
import aiir.dialects.async_dialect.passes
from aiir.passmanager import *


def run(f):
    print("\nTEST:", f.__name__)
    f()


# CHECK-LABEL: TEST: testCreateGroupOp
@run
def testCreateGroupOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i32 = IntegerType.get_signless(32)
            group_size = arith.ConstantOp(i32, 4)
            async_dialect.create_group(group_size)
        # CHECK:         %0 = "arith.constant"() <{value = 4 : i32}> : () -> i32
        # CHECK:         %1 = "async.create_group"(%0) : (i32) -> !async.group
        print(module)

def testAsyncPass():
    with Context() as context:
        PassManager.parse("any(async-to-async-runtime)")
    print("SUCCESS")


# CHECK-LABEL: testAsyncPass
#       CHECK: SUCCESS
run(testAsyncPass)
