#===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===##
import os
import sys

from libcxx.test.config import Configuration as LibcxxConfiguration


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

    def configure_compile_flags(self):
        self.cxx.compile_flags += ['-DLIBCXXABI_NO_TIMER']
        self.cxx.compile_flags += ['-funwind-tables']
        if not self.get_lit_bool('enable_threads', True):
            self.cxx.compile_flags += ['-DLIBCXXABI_HAS_NO_THREADS=1']
        super(Configuration, self).configure_compile_flags()    
    
    def configure_compile_flags_header_includes(self):
        self.configure_config_site_header()
        cxx_headers = self.get_lit_conf(
            'cxx_headers',
            os.path.join(self.libcxx_src_root, '/include'))
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

    def configure_compile_flags_exceptions(self):
        pass

    def configure_compile_flags_rtti(self):
        pass

    # TODO(ericwf): Remove this. This is a hack for OS X.
    # libc++ *should* export all of the symbols found in libc++abi on OS X.
    # For this reason LibcxxConfiguration will not link libc++abi in OS X.
    # However __cxa_throw_bad_new_array_length doesn't get exported into libc++
    # yet so we still need to explicitly link libc++abi.
    # See PR22654.
    def configure_link_flags_abi_library(self):
        self.cxx.link_flags += ['-lc++abi']

