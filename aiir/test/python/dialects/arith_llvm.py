# RUN: %PYTHON %s | FileCheck %s
from functools import partialmethod

from aiir.ir import *
import aiir.dialects.arith as arith
import aiir.dialects.func as func
import aiir.dialects.llvm as llvm


def run(f):
    print("\nTEST:", f.__name__)
    f()


# CHECK-LABEL: TEST: testOverflowFlags
# Test mostly to repro and verify error addressed for Python bindings.
@run
def testOverflowFlags():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = arith.ConstantOp(value=42, result=IntegerType.get_signless(32))
            r = arith.AddIOp(a, a, overflowFlags=arith.IntegerOverflowFlags.nsw)
            # CHECK: arith.addi {{.*}}, {{.*}} overflow<nsw> : i32
            print(r)
