#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

from libcxx.test.dsl import *
import sys

features = [
  Feature(name='fcoroutines-ts', compileFlag='-fcoroutines-ts',
          when=lambda cfg: hasCompileFlag(cfg, '-fcoroutines-ts') and
                           featureTestMacros(cfg, flags='-fcoroutines-ts').get('__cpp_coroutines', 0) >= 201703),

  Feature(name='thread-safety',                 when=lambda cfg: hasCompileFlag(cfg, '-Werror=thread-safety'), compileFlag='-Werror=thread-safety'),
  Feature(name='has-fblocks',                   when=lambda cfg: hasCompileFlag(cfg, '-fblocks')),
  Feature(name='-fsized-deallocation',          when=lambda cfg: hasCompileFlag(cfg, '-fsized-deallocation')),
  Feature(name='-faligned-allocation',          when=lambda cfg: hasCompileFlag(cfg, '-faligned-allocation')),
  Feature(name='fdelayed-template-parsing',     when=lambda cfg: hasCompileFlag(cfg, '-fdelayed-template-parsing')),
  Feature(name='libcpp-no-if-constexpr',        when=lambda cfg: '__cpp_if_constexpr' not in featureTestMacros(cfg)),
  Feature(name='libcpp-no-structured-bindings', when=lambda cfg: '__cpp_structured_bindings' not in featureTestMacros(cfg)),
  Feature(name='libcpp-no-deduction-guides',    when=lambda cfg: featureTestMacros(cfg).get('__cpp_deduction_guides', 0) < 201611),
  Feature(name='libcpp-no-concepts',            when=lambda cfg: featureTestMacros(cfg).get('__cpp_concepts', 0) < 201811),
  Feature(name='has-fobjc-arc',                 when=lambda cfg: hasCompileFlag(cfg, '-xobjective-c++ -fobjc-arc') and
                                                                 sys.platform.lower().strip() == 'darwin'), # TODO: this doesn't handle cross-compiling to Apple platforms.
  Feature(name='objective-c++',                 when=lambda cfg: hasCompileFlag(cfg, '-xobjective-c++ -fobjc-arc')),
  Feature(name='diagnose-if-support',           when=lambda cfg: hasCompileFlag(cfg, '-Wuser-defined-warnings'), compileFlag='-Wuser-defined-warnings'),
]
