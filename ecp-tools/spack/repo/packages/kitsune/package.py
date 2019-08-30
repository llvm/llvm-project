# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *
from spack.directives import *
import llnl.util.tty as tty

def build_variant_helpers(default_dict):
    """Build helper variant and project functions

    This returns two functions itself, closing over the dictionary passed as a way
    of faking a convenient static class variable that can still be used to define
    variants, dependencies and conflicts.
    """

    def _kitsune_variant(name, description, project=False, default=False,
                     values=None, deps={}, cmake_args=[]):
        """Make an LLVM variant
        
        Similar to `variant()` but allows more information to be associated with the
        variant in one place.  This function only exists to ensure the presence of
        default values for keys in `llvm_variants`.

        Args:

        name (str): Name of the variant
        description (str): Description of the variant
        project (bool): Whether this represents a project in the monorepo
        default(bool): Whether to enable this variant by default
        values (tuple or callable): same as spack `variant`
        deps (dict): Dict of dependencies, with the key being the dependency and
          the values being kwargs passed to `depends_on` merged with 
          when='+<name>'
        cmake_args (list): list of extra args to pass to cmake when this variant
          is enabled
    """
        if name in default_dict:
                raise ValueError(
                    '{} is already declared as a Kitsune variant'.format(name))
        
        default_dict[name] = {
            'description' : description,
            'project' : project,
            'default' : default,
            'values' : values,
            'deps' : deps,
            'cmake_args' : cmake_args,
        }

    def _kitsune_project(name, desc, **kwargs):
        """Define a variant that corresponds to an LLVM project

        Unless otherwise specified, this enables the variant (default=True), and
        ensures that the variant name is added to the LLVM_ENABLE_PROJECTS CMake
        variable
        """

        if 'default' not in kwargs:
            _kitsune_variant(name, desc, project=True, default=True, **kwargs)
        else:
            _kitsune_variant(name, desc, project=True, **kwargs)
    return _kitsune_variant, _kitsune_project


class Kitsune(CMakePackage):
    """Kitsune is a fork of LLVM that enables optimization within on-node parallel
    constructs by replacing opaque runtime function calls with true parallel
    entities at the LLVM IR level.
    """

    homepage = 'https://github.com/lanl/kitsune'
    url = 'https://github.com/lanl/kitsune/archive/kitsune-0.8.0.tar.gz'
    family = 'compiler'  # Used by lmod
    git = 'https://github.com/lanl/kitsune.git'

    version('0.8.0', tag='kitsune-0.8.0')
    version('develop', branch='release/8.x')


    # will hold all variants defined by kitsune_variant()
    kitsune_variants={}

    kitsune_variant, kitsune_project = build_variant_helpers(kitsune_variants)


    kitsune_variant(
        'shared_libs',
        'Build all components as shared libraries, faster, less memory to build,\
        less stable',
        cmake_args=['-DBUILD_SHARED_LIBS:Bool=ON']
    )

    kitsune_variant(
        'link_dylib',
        'Build and link the libLLVM shared library rather than static',
        cmake_args=['-DLLVM_LINK_LLVM_DYLIB:Bool=ON']
    )

    kitsune_variant(
        'all_targets',
        'Build all supported targets, default targets <current arch>,NVPTX,AMDGPU,CppBackend',
        cmake_args=['-DBUILD_SHARED_LIBS:Bool=ON']
        # NOTE: (probably?) Can't specify LLVM_TARGETS_TO_BUILD here because
        # we need a handle on spec.architecure.target
        # TODO: Test this
    )

    kitsune_variant(
        'build_type',
        'CMake build type',
        default='Release',
        values=('Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel')
        # NOTE: The debug version of LLVM is an order of magnitude larger than
        # the release version, and may take up 20-30 GB of space. If you want
        # to save space, build with `build_type=Release`.
    )

    kitsune_variant(
        'python',
        'Install python bindings',
        deps={'python':{}}
    )

    kitsune_variant(
        'gold',
        'Add support for LTO with the gold linker plugin',
        default=True,
        deps={'binutils+gold':{}}        
    )

 
    # Each project is one that can be enabled in the LLVM_ENABLE_PROJECTS CMake
    # var and has the effect of including projects defined in the top-level of
    # the Kitsune project. This does not necessarily imply they are part of
    # LLVM_ALL_PROJECTS (referenced with -DLLVM_ENABLE_PROJECTS=all) which has
    # been a moving target during the monorepo transition.

    kitsune_project(
        'clang',
        'Build the LLVM C/C++/Objective-C compiler frontend',
    )

    kitsune_project(
        'clang-tools-extra',
        'Build extra Clang-based tools (clangd, clang-tidy, etc.)',
    )

    kitsune_project(
        'libcxx',
        'Build the LLVM C++ standard library',
        cmake_args=['-DCLANG_DEFAULT_CXX_STDLIB=libc++']
    )

    kitsune_project(
        'libcxxabi',
        'Build the LLVM C++ ABI library',
    )

    kitsune_project(
        'libunwind',
        'Build the libcxxabi libunwind',
    )

    kitsune_project(
        'lldb',
        'Build the LLVM debugger',
        deps={'ncurses':{}, 'swig':{}, 'libedit':{}}
    )

    kitsune_project(
        'compiler-rt',
        'Build LLVM compiler runtime, including sanitizers',
    )

    kitsune_project(
        'lld',
        'Build the LLVM linker',
    )

    kitsune_project(
        'polly',
        'Build the LLVM polyhedral optimization plugin',
        cmake_args=["-DLLVM_LINK_POLLY_INTO_TOOLS:Bool=ON"]
    )

    kitsune_project(
        'debuginfo-tests',
        'Build tests for checking debug info generated by clang'
    )

    kitsune_project(
        'openmp',
        'Build LLVM\'s libomp'
    )



    # Actually make the variant and dependency declarations here
    for v_name, v_dict in kitsune_variants.items():
        v_deps       = v_dict['deps']
        v_vals       = v_dict['values']
        v_desc       = v_dict['description']
        v_default    = v_dict['default']

        variant(v_name,
                description = v_desc,
                default     = v_default,
                values      = v_vals)

        for dep, args in v_deps.items():
            # pop this out separately, because we may need to modify it
            when_arg = args.pop('when','')

            # When dealing with the dependency of a toggle-type variant
            # (+variant, ~variant), `when='+variant'` is implicit, so we build
            # it manually along with any other `when` spec in the dependency's
            # dictionary
            if v_default == True or v_default == False:
                when_arg = '+{}{}'.format(v_name, when_arg)

            # make the dependency declaration here, including anything else in
            # the dictionary
            depends_on(dep, when=when_arg, **args)


    # mapping of Spack spec architecture to (lower-case) the corresponding LLVM
    # target (as used in the CMake var LLVM_TARGETS_TO_BUILD)
    target_arch_mapping = {
        'x86'     : 'X86',
        'arm'     : 'ARM',
        'aarch64' : 'AArch64',
        'sparc'   : 'Sparc',
        'ppc'     : 'PowerPC',
        'power'   : 'PowerPC',
    }

    # TODO: make this conflict with project variants so
    #
    # This variant roughly corresponds in intent to -DLLVM_ENABLE_PROJECTS=all,
    # with the additional CMake variables each enabled project variant defines
    # variant('default_projects',
    #         default=True,
    #         "Enable a reasonable default set of (i.e. most) LLVM subprojects")
    
    
    extends('python', when='+python')

    
    # *** dependencies
    
    # NOTE: if libclc project is added, need to require 3.9.2+ here
    depends_on('cmake@3.4.3:', type='build')

    # even if we're not installing python bindings, we still need python to build
    depends_on('python', when='~python', type='build')

    # openmp uses Data::Dumper in openmp/runtime/tools/lib/tools.pm
    depends_on('perl-data-dumper', when="+openmp", type='build')

    # We only need this when both lldb and python variants are specified, so
    # this dependency must be specified manually (rather than as a 'deps' fields
    # of the variants dict)
    depends_on('py-six', when='+lldb +python')

    # conflicts('+lldb',        when='~clang')

    @run_before('cmake')
    def check_darwin_lldb_codesign_requirement(self):
        if not self.spec.satisfies('+lldb platform=darwin'):
            return
        codesign = which('codesign')
        mkdir('tmp')
        llvm_check_file = join_path('tmp', 'llvm_check')
        copy('/usr/bin/false', llvm_check_file)

        try:
            codesign('-f', '-s', 'lldb_codesign', '--dryrun',
                     llvm_check_file)

        except ProcessError:
            explanation = ('The "lldb_codesign" identity must be available'
                           ' to build LLVM with LLDB. See https://llvm.org/'
                           'svn/llvm-project/lldb/trunk/docs/code-signing'
                           '.txt for details on how to create this identity.')
            raise RuntimeError(explanation)

    
    def setup_environment(self, spack_env, run_env):
        # set the appropriate c++11 flag in the build environment for whatever
        # compiler is being used
        spack_env.append_flags('CXXFLAGS', self.compiler.cxx11_flag)

        # environment vars set in the modulefile
        if '+clang' in self.spec:
            run_env.set('CC', join_path(self.spec.prefix.bin, 'clang'))
            run_env.set('CXX', join_path(self.spec.prefix.bin, 'clang++'))


    # With the new LLVM monorepo, CMakeLists.txt lives in the llvm subdirectory.
    @property
    def root_cmakelists_dir(self):
        """The relative path to the directory containing CMakeLists.txt

        This path is relative to the root of the extracted tarball,
        not to the ``build_directory``. Defaults to the current directory.

        :return: directory containing CMakeLists.txt
        """
        return 'llvm'



    def cmake_args(self):
        spec = self.spec
        cmake_args = [
            '-DLLVM_REQUIRES_RTTI:BOOL=ON',
            '-DLLVM_ENABLE_RTTI:BOOL=ON',
            '-DLLVM_ENABLE_EH:BOOL=ON',
            '-DCLANG_DEFAULT_OPENMP_RUNTIME:STRING=libomp',
            '-DPYTHON_EXECUTABLE:PATH={0}'.format(spec['python'].command.path),
        ]

        # TODO: Instead of unconditionally disabling CUDA, add a "cuda" variant
        #       (see TODO in llvm spack package), and set the paths if enabled.
        cmake_args.extend([
            '-DCUDA_TOOLKIT_ROOT_DIR:PATH=IGNORE',
            '-DCUDA_SDK_ROOT_DIR:PATH=IGNORE',
            '-DCUDA_NVCC_EXECUTABLE:FILEPATH=IGNORE',
            '-DLIBOMPTARGET_DEP_CUDA_DRIVER_LIBRARIES:STRING=IGNORE'])

        enable_projects=[]
        for vrnt_name, vrnt_descr in variants.items():
            variant_on = '+{}'.format(vrnt_name) in spec
            variant_off = '~{}'.format(vrnt_name) in spec
            is_project = vrnt_descr.get('project')

            if variant_on:
                cmake_args.extend(vrnt_descr.get('cmake_args'))
                
                if is_project:
                    # We have to enable each sub-project one by one, even if all
                    # are enabled, since the 'all' alias for
                    # LLVM_ENABLE_PROJECTS isn't reliable
                    enabled_projects.append(vrnt_name)
                    if '+all_projects' in spec:
                        tty.warn('+{} is redudant with +all_projects'.format(vrnt_name))
                        
        
        cmake_args.append(
            '-DLLVM_ENABLE_PROJECTS={}'.format(';'.join(enabled_projects)))
    
        
        # I'm pretty sure this has to be computed here, rather than in the
        # variants declaration, since it depends on the binutils dependency spec
        if '+gold' in spec:
            cmake_args.append('-DLLVM_BINUTILS_INCDIR=' +
                              spec['binutils'].prefix.include)

        # since this depends on two variants, it also cannot be expressed in the
        # variant dict
        if '+python' in spec and '+lldb' in spec:
            cmake_args.append('-DLLDB_USE_SYSTEM_SIX:Bool=TRUE')


        if '+all_targets' not in spec:  # all is default on cmake

            targets = ['NVPTX', 'AMDGPU']
            this_arch = spec.architecture.target
            this_target = target_arch_mapping.get(this_arch.lower())
            if this_target:
                targets.append(this_target)
            else:
                tty.warn("Target {} not identified as supported by LLVM".format(this_arch))

            cmake_args.append(
                '-DLLVM_TARGETS_TO_BUILD:STRING=' + ';'.join(targets))

        if spec.satisfies('platform=linux'):
            # set the RPATH to the install path at build time (rather than
            # relinking at install time)
            cmake_args.append('-DCMAKE_BUILD_WITH_INSTALL_RPATH=1')
        return cmake_args

    @run_before('build')
    def pre_install(self):
        with working_dir(self.build_directory):
            # When building shared libraries these need to be installed first
            make('install-LLVMTableGen')
            make('install-LLVMDemangle')
            make('install-LLVMSupport')

    @run_after('install')
    def post_install(self):
        if '+clang' in self.spec and '+python' in self.spec:
            install_tree(
                'tools/clang/bindings/python/clang',
                join_path(site_packages_dir, 'clang'))

        with working_dir(self.build_directory):
            install_tree('bin', join_path(self.prefix, 'libexec', 'llvm'))
