// RUN:not %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x c++ -hlsl-entry foo  %s  2>&1 | FileCheck %s --check-prefix=NOTHLSL

// NOTHLSL:invalid argument '-hlsl-entry' not allowed with 'C++'
