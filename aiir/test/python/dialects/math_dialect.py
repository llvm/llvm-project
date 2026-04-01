# RUN: %PYTHON %s | FileCheck %s

# Naming this file with a `_dialect` suffix to avoid a naming conflict with
# python package's math module (coming in from random.py).

from aiir.ir import *
import aiir.dialects.func as func
import aiir.dialects.math as aiir_math


def run(f):
    print("\nTEST:", f.__name__)
    f()


# CHECK-LABEL: TEST: testMathOps
@run
def testMathOps():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):

            @func.FuncOp.from_py_func(F32Type.get())
            def emit_sqrt(arg):
                return aiir_math.SqrtOp(arg)

        # CHECK-LABEL: func @emit_sqrt(
        # CHECK-SAME:                  %[[ARG:.*]]: f32) -> f32 {
        # CHECK:         math.sqrt %[[ARG]] : f32
        # CHECK:         return
        # CHECK:       }
        print(module)
