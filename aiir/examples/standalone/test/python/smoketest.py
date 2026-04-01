# RUN: %python %s 2>&1 | FileCheck %s
import sys

# CHECK: Testing aiir_standalone package
print("Testing aiir_standalone package", file=sys.stderr)

import aiir_standalone.ir
from aiir_standalone.dialects import standalone_nanobind as standalone_d

with aiir_standalone.ir.Context():
    standalone_d.register_dialects()
    standalone_module = aiir_standalone.ir.Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = standalone.foo %0 : i32
    """
    )
    # CHECK: %[[C2:.*]] = arith.constant 2 : i32
    # CHECK: standalone.foo %[[C2]] : i32
    print(str(standalone_module), file=sys.stderr)

    custom_type = standalone_d.CustomType.get("foo")
    # CHECK: !standalone.custom<"foo">
    print(custom_type, file=sys.stderr)

    # CHECK: this is a fp16 type
    standalone_d.print_fp_type(aiir_standalone.ir.F16Type.get(), sys.stderr)
    # CHECK: this is a fp32 type
    standalone_d.print_fp_type(aiir_standalone.ir.F32Type.get(), sys.stderr)
    # CHECK: this is a fp64 type
    standalone_d.print_fp_type(aiir_standalone.ir.F64Type.get(), sys.stderr)


# CHECK: Testing aiir package
print("Testing aiir package", file=sys.stderr)

from aiir.ir import *

# CHECK-NOT: RuntimeWarning: nanobind: type '{{.*}}' was already registered!
