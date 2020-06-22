#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##
import os
import sys

from libcxx.test.config import Configuration as LibcxxConfiguration
from libcxx.test.config import intMacroValue


class Configuration(LibcxxConfiguration):
    # pylint: disable=redefined-outer-name
    def __init__(self, lit_config, config):
        super(Configuration, self).__init__(lit_config, config)
        self.libcxxabi_src_root = None
        self.libcxxabi_obj_root = None
        self.abi_library_path = None
        self.libcxx_src_root = None

    def configure_src_root(self):
        self.libcxxabi_src_root = self.get_lit_conf(
            'libcxxabi_src_root',
            os.path.dirname(self.config.test_source_root))
        self.libcxx_src_root = self.get_lit_conf(
            'libcxx_src_root',
            os.path.join(self.libcxxabi_src_root, '/../libcxx'))

    def configure_obj_root(self):
        self.libcxxabi_obj_root = self.get_lit_conf('libcxxabi_obj_root')
        super(Configuration, self).configure_obj_root()

    def has_cpp_feature(self, feature, required_value):
        return intMacroValue(self.cxx.dumpMacros().get('__cpp_' + feature, '0')) >= required_value

    def configure_features(self):
        super(Configuration, self).configure_features()
        if not self.has_cpp_feature('noexcept_function_type', 201510):
            self.config.available_features.add('libcxxabi-no-noexcept-function-type')
        if not self.get_lit_bool('llvm_unwinder', False):
            self.config.available_features.add('libcxxabi-has-system-unwinder')

    def configure_compile_flags(self):
        self.cxx.compile_flags += [
            '-DLIBCXXABI_NO_TIMER',
            '-D_LIBCPP_ENABLE_CXX17_REMOVED_UNEXPECTED_FUNCTIONS',
        ]
        if self.get_lit_bool('enable_exceptions', True):
            self.cxx.compile_flags += ['-funwind-tables']
        if not self.get_lit_bool('enable_threads', True):
            self.cxx.compile_flags += ['-D_LIBCXXABI_HAS_NO_THREADS']
            self.config.available_features.add('libcxxabi-no-threads')
        super(Configuration, self).configure_compile_flags()

    def configure_compile_flags_header_includes(self):
        self.configure_config_site_header()
        cxx_headers = self.get_lit_conf(
            'cxx_headers',
            os.path.join(self.libcxx_src_root, '/include'))
        if cxx_headers == '':
            self.lit_config.note('using the systems c++ headers')
        else:
            self.cxx.compile_flags += ['-nostdinc++']
        if not os.path.isdir(cxx_headers):
            self.lit_config.fatal("cxx_headers='%s' is not a directory."
                                  % cxx_headers)
        self.cxx.compile_flags += ['-I' + cxx_headers]

        libcxxabi_headers = self.get_lit_conf(
            'libcxxabi_headers',
            os.path.join(self.libcxxabi_src_root, 'include'))
        if not os.path.isdir(libcxxabi_headers):
            self.lit_config.fatal("libcxxabi_headers='%s' is not a directory."
                                  % libcxxabi_headers)
        self.cxx.compile_flags += ['-I' + libcxxabi_headers]

        libunwind_headers = self.get_lit_conf('libunwind_headers', None)
        if self.get_lit_bool('llvm_unwinder', False) and libunwind_headers:
            if not os.path.isdir(libunwind_headers):
                self.lit_config.fatal("libunwind_headers='%s' is not a directory."
                                      % libunwind_headers)
            self.cxx.compile_flags += ['-I' + libunwind_headers]

    def configure_compile_flags_rtti(self):
        pass
