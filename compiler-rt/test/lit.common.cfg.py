# -*- Python -*-

# Configuration file for 'lit' test runner.
# This file contains common rules for various compiler-rt testsuites.
# It is mostly copied from lit.cfg.py used by Clang.
import os
import platform
import re
import subprocess
import json

import lit.formats
import lit.util

# Get shlex.quote if available (added in 3.3), and fall back to pipes.quote if
# it's not available.
try:
  import shlex
  sh_quote = shlex.quote
except:
  import pipes
  sh_quote = pipes.quote

def find_compiler_libdir():
  """
    Returns the path to library resource directory used
    by the compiler.
  """
  if config.compiler_id != 'Clang':
    lit_config.warning(f'Determining compiler\'s runtime directory is not supported for {config.compiler_id}')
    # TODO: Support other compilers.
    return None
  def get_path_from_clang(args, allow_failure):
    clang_cmd = [
      config.clang.strip(),
      f'--target={config.target_triple}',
    ]
    clang_cmd.extend(args)
    path = None
    try:
      result = subprocess.run(
        clang_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
      )
      path = result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
      msg = f'Failed to run {clang_cmd}\nrc:{e.returncode}\nstdout:{e.stdout}\ne.stderr{e.stderr}'
      if allow_failure:
        lit_config.warning(msg)
      else:
        lit_config.fatal(msg)
    return path, clang_cmd

  # Try using `-print-runtime-dir`. This is only supported by very new versions of Clang.
  # so allow failure here.
  runtime_dir, clang_cmd = get_path_from_clang(shlex.split(config.target_cflags)
                                               + ['-print-runtime-dir'],
                                               allow_failure=True)
  if runtime_dir:
    if os.path.exists(runtime_dir):
      return os.path.realpath(runtime_dir)
    # TODO(dliew): This should be a fatal error but it seems to trip the `llvm-clang-win-x-aarch64`
    # bot which is likely misconfigured
    lit_config.warning(
      f'Path reported by clang does not exist: \"{runtime_dir}\". '
      f'This path was found by running {clang_cmd}.'
    )
    return None

  # Fall back for older AppleClang that doesn't support `-print-runtime-dir`
  # Note `-print-file-name=<path to compiler-rt lib>` was broken for Apple
  # platforms so we can't use that approach here (see https://reviews.llvm.org/D101682).
  if config.host_os == 'Darwin':
    lib_dir, _ = get_path_from_clang(['-print-file-name=lib'], allow_failure=False)
    runtime_dir = os.path.join(lib_dir, 'darwin')
    if not os.path.exists(runtime_dir):
      lit_config.fatal(f'Path reported by clang does not exist: {runtime_dir}')
    return os.path.realpath(runtime_dir)

  lit_config.warning('Failed to determine compiler\'s runtime directory')
  return None


# Choose between lit's internal shell pipeline runner and a real shell.  If
# LIT_USE_INTERNAL_SHELL is in the environment, we use that as an override.
use_lit_shell = os.environ.get("LIT_USE_INTERNAL_SHELL")
if use_lit_shell:
    # 0 is external, "" is default, and everything else is internal.
    execute_external = (use_lit_shell == "0")
else:
    # Otherwise we default to internal on Windows and external elsewhere, as
    # bash on Windows is usually very slow.
    execute_external = (not sys.platform in ['win32'])

# Allow expanding substitutions that are based on other substitutions
config.recursiveExpansionLimit = 10

# Setup test format.
config.test_format = lit.formats.ShTest(execute_external)
if execute_external:
  config.available_features.add('shell')

target_is_msvc = bool(re.match(r'.*-windows-msvc$', config.target_triple))

compiler_id = getattr(config, 'compiler_id', None)
if compiler_id == "Clang":
  if not (platform.system() == 'Windows' and target_is_msvc):
    config.cxx_mode_flags = ["--driver-mode=g++"]
  else:
    config.cxx_mode_flags = []
  # We assume that sanitizers should provide good enough error
  # reports and stack traces even with minimal debug info.
  config.debug_info_flags = ["-gline-tables-only"]
  if platform.system() == 'Windows' and target_is_msvc:
    # On MSVC, use CodeView with column info instead of DWARF. Both VS and
    # windbg do not behave well when column info is enabled, but users have
    # requested it because it makes ASan reports more precise.
    config.debug_info_flags.append("-gcodeview")
    config.debug_info_flags.append("-gcolumn-info")
elif compiler_id == 'GNU':
  config.cxx_mode_flags = ["-x c++"]
  config.debug_info_flags = ["-g"]
else:
  lit_config.fatal("Unsupported compiler id: %r" % compiler_id)
# Add compiler ID to the list of available features.
config.available_features.add(compiler_id)

# When LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=on, the initial value of
# config.compiler_rt_libdir (COMPILER_RT_RESOLVED_LIBRARY_OUTPUT_DIR) has the
# triple as the trailing path component. The value is incorrect for -m32/-m64.
# Adjust config.compiler_rt accordingly.
if config.enable_per_target_runtime_dir:
    if '-m32' in shlex.split(config.target_cflags):
        config.compiler_rt_libdir = re.sub(r'/x86_64(?=-[^/]+$)', '/i386', config.compiler_rt_libdir)
    elif '-m64' in shlex.split(config.target_cflags):
        config.compiler_rt_libdir = re.sub(r'/i386(?=-[^/]+$)', '/x86_64', config.compiler_rt_libdir)

# Ask the compiler for the path to libraries it is going to use. If this
# doesn't match config.compiler_rt_libdir then it means we might be testing the
# compiler's own runtime libraries rather than the ones we just built.
# Warn about about this and handle appropriately.
compiler_libdir = find_compiler_libdir()
if compiler_libdir:
  compiler_rt_libdir_real = os.path.realpath(config.compiler_rt_libdir)
  if compiler_libdir != compiler_rt_libdir_real:
    lit_config.warning(
      'Compiler lib dir != compiler-rt lib dir\n'
      f'Compiler libdir:     "{compiler_libdir}"\n'
      f'compiler-rt libdir:  "{compiler_rt_libdir_real}"')
    if config.test_standalone_build_libs:
      # Use just built runtime libraries, i.e. the the libraries this built just built.
      if not config.test_suite_supports_overriding_runtime_lib_path:
        # Test suite doesn't support this configuration.
        # TODO(dliew): This should be an error but it seems several bots are
        # testing incorrectly and having this as an error breaks them.
        lit_config.warning(
            'COMPILER_RT_TEST_STANDALONE_BUILD_LIBS=ON, but this test suite '
            'does not support testing the just-built runtime libraries '
            'when the test compiler is configured to use different runtime '
            'libraries. Either modify this test suite to support this test '
            'configuration, or set COMPILER_RT_TEST_STANDALONE_BUILD_LIBS=OFF '
            'to test the runtime libraries included in the compiler instead.'
        )
    else:
      # Use Compiler's resource library directory instead.
      config.compiler_rt_libdir = compiler_libdir
    lit_config.note(f'Testing using libraries in "{config.compiler_rt_libdir}"')

# If needed, add cflag for shadow scale.
if config.asan_shadow_scale != '':
  config.target_cflags += " -mllvm -asan-mapping-scale=" + config.asan_shadow_scale
if config.memprof_shadow_scale != '':
  config.target_cflags += " -mllvm -memprof-mapping-scale=" + config.memprof_shadow_scale

# Clear some environment variables that might affect Clang.
possibly_dangerous_env_vars = ['ASAN_OPTIONS', 'DFSAN_OPTIONS', 'HWASAN_OPTIONS',
                               'LSAN_OPTIONS', 'MSAN_OPTIONS', 'UBSAN_OPTIONS',
                               'COMPILER_PATH', 'RC_DEBUG_OPTIONS',
                               'CINDEXTEST_PREAMBLE_FILE', 'LIBRARY_PATH',
                               'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH',
                               'OBJC_INCLUDE_PATH', 'OBJCPLUS_INCLUDE_PATH',
                               'LIBCLANG_TIMING', 'LIBCLANG_OBJTRACKING',
                               'LIBCLANG_LOGGING', 'LIBCLANG_BGPRIO_INDEX',
                               'LIBCLANG_BGPRIO_EDIT', 'LIBCLANG_NOTHREADS',
                               'LIBCLANG_RESOURCE_USAGE',
                               'LIBCLANG_CODE_COMPLETION_LOGGING',
                               'XRAY_OPTIONS']
# Clang/MSVC may refer to %INCLUDE%. vsvarsall.bat sets it.
if not (platform.system() == 'Windows' and target_is_msvc):
    possibly_dangerous_env_vars.append('INCLUDE')
for name in possibly_dangerous_env_vars:
  if name in config.environment:
    del config.environment[name]

# Tweak PATH to include llvm tools dir.
if (not config.llvm_tools_dir) or (not os.path.exists(config.llvm_tools_dir)):
  lit_config.fatal("Invalid llvm_tools_dir config attribute: %r" % config.llvm_tools_dir)
path = os.path.pathsep.join((config.llvm_tools_dir, config.environment['PATH']))
config.environment['PATH'] = path

# Help MSVS link.exe find the standard libraries.
# Make sure we only try to use it when targetting Windows.
if platform.system() == 'Windows' and target_is_msvc:
  config.environment['LIB'] = os.environ['LIB']

config.available_features.add(config.host_os.lower())

if config.target_triple.startswith("ppc") or config.target_triple.startswith("powerpc"):
  config.available_features.add("ppc")

if re.match(r'^x86_64.*-linux', config.target_triple):
  config.available_features.add("x86_64-linux")

if lit.util.isMacOSTriple(config.target_triple):
   config.available_features.add('darwin')

config.available_features.add("host-byteorder-" + sys.byteorder + "-endian")

if config.have_zlib == "1":
  config.available_features.add("zlib")

# Use ugly construction to explicitly prohibit "clang", "clang++" etc.
# in RUN lines.
config.substitutions.append(
    (' clang', """\n\n*** Do not use 'clangXXX' in tests,
     instead define '%clangXXX' substitution in lit config. ***\n\n""") )

if config.host_os == 'NetBSD':
  nb_commands_dir = os.path.join(config.compiler_rt_src_root,
                                 "test", "sanitizer_common", "netbsd_commands")
  config.netbsd_noaslr_prefix = ('sh ' +
                                 os.path.join(nb_commands_dir, 'run_noaslr.sh'))
  config.netbsd_nomprotect_prefix = ('sh ' +
                                     os.path.join(nb_commands_dir,
                                                  'run_nomprotect.sh'))
  config.substitutions.append( ('%run_nomprotect',
                                config.netbsd_nomprotect_prefix) )
else:
  config.substitutions.append( ('%run_nomprotect', '%run') )

# Copied from libcxx's config.py
def get_lit_conf(name, default=None):
    # Allow overriding on the command line using --param=<name>=<val>
    val = lit_config.params.get(name, None)
    if val is None:
        val = getattr(config, name, None)
        if val is None:
            val = default
    return val

emulator = get_lit_conf('emulator', None)

def get_ios_commands_dir():
  return os.path.join(config.compiler_rt_src_root, "test", "sanitizer_common", "ios_commands")

# Allow tests to be executed on a simulator or remotely.
if emulator:
  config.substitutions.append( ('%run', emulator) )
  config.substitutions.append( ('%env ', "env ") )
  # TODO: Implement `%device_rm` to perform removal of files in the emulator.
  # For now just make it a no-op.
  lit_config.warning('%device_rm is not implemented')
  config.substitutions.append( ('%device_rm', 'echo ') )
  config.compile_wrapper = ""
elif config.host_os == 'Darwin' and config.apple_platform != "osx":
  # Darwin tests can be targetting macOS, a device or a simulator. All devices
  # are declared as "ios", even for iOS derivatives (tvOS, watchOS). Similarly,
  # all simulators are "iossim". See the table below.
  #
  # =========================================================================
  # Target             | Feature set
  # =========================================================================
  # macOS              | darwin
  # iOS device         | darwin, ios
  # iOS simulator      | darwin, ios, iossim
  # tvOS device        | darwin, ios, tvos
  # tvOS simulator     | darwin, ios, iossim, tvos, tvossim
  # watchOS device     | darwin, ios, watchos
  # watchOS simulator  | darwin, ios, iossim, watchos, watchossim
  # =========================================================================

  ios_or_iossim = "iossim" if config.apple_platform.endswith("sim") else "ios"

  config.available_features.add('ios')
  device_id_env = "SANITIZER_" + ios_or_iossim.upper() + "_TEST_DEVICE_IDENTIFIER"
  if ios_or_iossim == "iossim":
    config.available_features.add('iossim')
    if device_id_env not in os.environ:
      lit_config.fatal(
        '{} must be set in the environment when running iossim tests'.format(
          device_id_env))
  if config.apple_platform != "ios" and config.apple_platform != "iossim":
    config.available_features.add(config.apple_platform)

  ios_commands_dir = get_ios_commands_dir()

  run_wrapper = os.path.join(ios_commands_dir, ios_or_iossim + "_run.py")
  env_wrapper = os.path.join(ios_commands_dir, ios_or_iossim + "_env.py")
  compile_wrapper = os.path.join(ios_commands_dir, ios_or_iossim + "_compile.py")
  prepare_script = os.path.join(ios_commands_dir, ios_or_iossim + "_prepare.py")

  if device_id_env in os.environ:
    config.environment[device_id_env] = os.environ[device_id_env]
  config.substitutions.append(('%run', run_wrapper))
  config.substitutions.append(('%env ', env_wrapper + " "))
  # Current implementation of %device_rm uses the run_wrapper to do
  # the work.
  config.substitutions.append(('%device_rm', '{} rm '.format(run_wrapper)))
  config.compile_wrapper = compile_wrapper

  try:
    prepare_output = subprocess.check_output([prepare_script, config.apple_platform, config.clang]).decode().strip()
  except subprocess.CalledProcessError as e:
    print("Command failed:")
    print(e.output)
    raise e
  if len(prepare_output) > 0: print(prepare_output)
  prepare_output_json = prepare_output.split("\n")[-1]
  prepare_output = json.loads(prepare_output_json)
  config.environment.update(prepare_output["env"])
elif config.android:
  config.available_features.add('android')
  compile_wrapper = os.path.join(config.compiler_rt_src_root, "test", "sanitizer_common", "android_commands", "android_compile.py") + " "
  config.compile_wrapper = compile_wrapper
  config.substitutions.append( ('%run', "") )
  config.substitutions.append( ('%env ', "env ") )
else:
  config.substitutions.append( ('%run', "") )
  config.substitutions.append( ('%env ', "env ") )
  # When running locally %device_rm is a no-op.
  config.substitutions.append( ('%device_rm', 'echo ') )
  config.compile_wrapper = ""

# Define CHECK-%os to check for OS-dependent output.
config.substitutions.append( ('CHECK-%os', ("CHECK-" + config.host_os)))

# Define %arch to check for architecture-dependent output.
config.substitutions.append( ('%arch', (config.host_arch)))

if config.host_os == 'Windows':
  # FIXME: This isn't quite right. Specifically, it will succeed if the program
  # does not crash but exits with a non-zero exit code. We ought to merge
  # KillTheDoctor and not --crash to make the latter more useful and remove the
  # need for this substitution.
  config.expect_crash = "not KillTheDoctor "
else:
  config.expect_crash = "not --crash "

config.substitutions.append( ("%expect_crash ", config.expect_crash) )

target_arch = getattr(config, 'target_arch', None)
if target_arch:
  config.available_features.add(target_arch + '-target-arch')
  if target_arch in ['x86_64', 'i386']:
    config.available_features.add('x86-target-arch')
  config.available_features.add(target_arch + '-' + config.host_os.lower())

compiler_rt_debug = getattr(config, 'compiler_rt_debug', False)
if not compiler_rt_debug:
  config.available_features.add('compiler-rt-optimized')

libdispatch = getattr(config, 'compiler_rt_intercept_libdispatch', False)
if libdispatch:
  config.available_features.add('libdispatch')

sanitizer_can_use_cxxabi = getattr(config, 'sanitizer_can_use_cxxabi', True)
if sanitizer_can_use_cxxabi:
  config.available_features.add('cxxabi')

if not getattr(config, 'sanitizer_uses_static_cxxabi', False):
  config.available_features.add('shared_cxxabi')

if not getattr(config, 'sanitizer_uses_static_unwind', False):
  config.available_features.add('shared_unwind')

if config.has_lld:
  config.available_features.add('lld-available')

if config.use_lld:
  config.available_features.add('lld')

if config.can_symbolize:
  config.available_features.add('can-symbolize')

if config.gwp_asan:
  config.available_features.add('gwp_asan')

lit.util.usePlatformSdkOnDarwin(config, lit_config)

min_macos_deployment_target_substitutions = [
  (10, 11),
  (10, 12),
]
# TLS requires watchOS 3+
config.substitutions.append( ('%darwin_min_target_with_tls_support', '%min_macos_deployment_target=10.12') )

if config.host_os == 'Darwin':
  osx_version = (10, 0, 0)
  try:
    osx_version = subprocess.check_output(["sw_vers", "-productVersion"],
                                          universal_newlines=True)
    osx_version = tuple(int(x) for x in osx_version.split('.'))
    if len(osx_version) == 2: osx_version = (osx_version[0], osx_version[1], 0)
    if osx_version >= (10, 11):
      config.available_features.add('osx-autointerception')
      config.available_features.add('osx-ld64-live_support')
    if osx_version >= (13, 1):
      config.available_features.add('jit-compatible-osx-swift-runtime')
  except subprocess.CalledProcessError:
    pass

  config.darwin_osx_version = osx_version

  # Detect x86_64h
  try:
    output = subprocess.check_output(["sysctl", "hw.cpusubtype"])
    output_re = re.match("^hw.cpusubtype: ([0-9]+)$", output)
    if output_re:
      cpu_subtype = int(output_re.group(1))
      if cpu_subtype == 8: # x86_64h
        config.available_features.add('x86_64h')
  except:
    pass

  # 32-bit iOS simulator is deprecated and removed in latest Xcode.
  if config.apple_platform == "iossim":
    if config.target_arch == "i386":
      config.unsupported = True

  def get_macos_aligned_version(macos_vers):
    platform = config.apple_platform
    if platform == 'osx':
      return macos_vers

    macos_major, macos_minor = macos_vers
    assert macos_major >= 10

    if macos_major == 10:  # macOS 10.x
      major = macos_minor
      minor = 0
    else:                  # macOS 11+
      major = macos_major + 5
      minor = macos_minor

    assert major >= 11

    if platform.startswith('ios') or platform.startswith('tvos'):
      major -= 2
    elif platform.startswith('watch'):
      major -= 9
    else:
      lit_config.fatal("Unsupported apple platform '{}'".format(platform))

    return (major, minor)

  for vers in min_macos_deployment_target_substitutions:
    flag = config.apple_platform_min_deployment_target_flag
    major, minor = get_macos_aligned_version(vers)
    if 'mtargetos' in flag:
      sim = '-simulator' if 'sim' in config.apple_platform else ''
      config.substitutions.append( ('%%min_macos_deployment_target=%s.%s' % vers, '{}{}.{}{}'.format(flag, major, minor, sim)) )
    else:
      config.substitutions.append( ('%%min_macos_deployment_target=%s.%s' % vers, '{}={}.{}'.format(flag, major, minor)) )
else:
  for vers in min_macos_deployment_target_substitutions:
    config.substitutions.append( ('%%min_macos_deployment_target=%s.%s' % vers, '') )

if config.android:
  env = os.environ.copy()
  if config.android_serial:
    env['ANDROID_SERIAL'] = config.android_serial
    config.environment['ANDROID_SERIAL'] = config.android_serial

  adb = os.environ.get('ADB', 'adb')

  # These are needed for tests to upload/download temp files, such as
  # suppression-files, to device.
  config.substitutions.append( ('%device_rundir/', "/data/local/tmp/Output/") )
  config.substitutions.append( ('%push_to_device', "%s -s '%s' push " % (adb, env['ANDROID_SERIAL']) ) )
  config.substitutions.append( ('%adb_shell ', "%s -s '%s' shell " % (adb, env['ANDROID_SERIAL']) ) )
  config.substitutions.append( ('%device_rm', "%s -s '%s' shell 'rm ' " % (adb, env['ANDROID_SERIAL']) ) )

  try:
    android_api_level_str = subprocess.check_output([adb, "shell", "getprop", "ro.build.version.sdk"], env=env).rstrip()
    android_api_codename = subprocess.check_output([adb, "shell", "getprop", "ro.build.version.codename"], env=env).rstrip().decode("utf-8")
  except (subprocess.CalledProcessError, OSError):
    lit_config.fatal("Failed to read ro.build.version.sdk (using '%s' as adb)" % adb)
  try:
    android_api_level = int(android_api_level_str)
  except ValueError:
    lit_config.fatal("Failed to read ro.build.version.sdk (using '%s' as adb): got '%s'" % (adb, android_api_level_str))
  android_api_level = min(android_api_level, int(config.android_api_level))
  for required in [26, 28, 29, 30]:
    if android_api_level >= required:
      config.available_features.add('android-%s' % required)
  # FIXME: Replace with appropriate version when availible.
  if android_api_level > 30 or (android_api_level == 30 and android_api_codename == 'S'):
    config.available_features.add('android-thread-properties-api')

  # Prepare the device.
  android_tmpdir = '/data/local/tmp/Output'
  subprocess.check_call([adb, "shell", "mkdir", "-p", android_tmpdir], env=env)
  for file in config.android_files_to_push:
    subprocess.check_call([adb, "push", file, android_tmpdir], env=env)
else:
  config.substitutions.append( ('%device_rundir/', "") )
  config.substitutions.append( ('%push_to_device', "echo ") )
  config.substitutions.append( ('%adb_shell', "echo ") )

if config.host_os == 'Linux':
  # detect whether we are using glibc, and which version
  # NB: 'ldd' is just one of the tools commonly installed as part of glibc/musl
  ldd_ver_cmd = subprocess.Popen(['ldd', '--version'],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL,
                                 env={'LANG': 'C'})
  sout, _ = ldd_ver_cmd.communicate()
  ver_lines = sout.splitlines()
  if not config.android and len(ver_lines) and ver_lines[0].startswith(b"ldd "):
    from distutils.version import LooseVersion
    ver = LooseVersion(ver_lines[0].split()[-1].decode())
    for required in ["2.27", "2.30", "2.34", "2.37"]:
      if ver >= LooseVersion(required):
        config.available_features.add("glibc-" + required)

sancovcc_path = os.path.join(config.llvm_tools_dir, "sancov")
if os.path.exists(sancovcc_path):
  config.available_features.add("has_sancovcc")
  config.substitutions.append( ("%sancovcc ", sancovcc_path) )

def liblto_path():
  return os.path.join(config.llvm_shlib_dir, 'libLTO.dylib')

def is_darwin_lto_supported():
  return os.path.exists(liblto_path())

def is_binutils_lto_supported():
  if not os.path.exists(os.path.join(config.llvm_shlib_dir, 'LLVMgold.so')):
    return False

  # We require both ld.bfd and ld.gold exist and support plugins. They are in
  # the same repository 'binutils-gdb' and usually built together.
  for exe in (config.gnu_ld_executable, config.gold_executable):
    try:
      ld_cmd = subprocess.Popen([exe, '--help'], stdout=subprocess.PIPE, env={'LANG': 'C'})
      ld_out = ld_cmd.stdout.read().decode()
      ld_cmd.wait()
    except OSError:
      return False
    if not '-plugin' in ld_out:
      return False

  return True

def is_windows_lto_supported():
  if not target_is_msvc:
    return True
  return os.path.exists(os.path.join(config.llvm_tools_dir, 'lld-link.exe'))

if config.host_os == 'Darwin' and is_darwin_lto_supported():
  config.lto_supported = True
  config.lto_flags = [ '-Wl,-lto_library,' + liblto_path() ]
elif config.host_os in ['Linux', 'FreeBSD', 'NetBSD']:
  config.lto_supported = False
  if config.use_lld:
    config.lto_supported = True
  if is_binutils_lto_supported():
    config.available_features.add('binutils_lto')
    config.lto_supported = True

  if config.lto_supported:
    if config.use_lld:
      config.lto_flags = ["-fuse-ld=lld"]
    else:
      config.lto_flags = ["-fuse-ld=gold"]
elif config.host_os == 'Windows' and is_windows_lto_supported():
  config.lto_supported = True
  config.lto_flags = ["-fuse-ld=lld"]
else:
  config.lto_supported = False

if config.lto_supported:
  config.available_features.add('lto')
  if config.use_thinlto:
    config.available_features.add('thinlto')
    config.lto_flags += ["-flto=thin"]
  else:
    config.lto_flags += ["-flto"]

if config.have_rpc_xdr_h:
  config.available_features.add('sunrpc')

# Ask llvm-config about assertion mode.
try:
  llvm_config_cmd = subprocess.Popen(
      [os.path.join(config.llvm_tools_dir, 'llvm-config'), '--assertion-mode'],
      stdout = subprocess.PIPE,
      env=config.environment)
except OSError as e:
  print("Could not launch llvm-config in " + config.llvm_tools_dir)
  print("    Failed with error #{0}: {1}".format(e.errno, e.strerror))
  exit(42)

if re.search(r'ON', llvm_config_cmd.stdout.read().decode('ascii')):
  config.available_features.add('asserts')
llvm_config_cmd.wait()

# Sanitizer tests tend to be flaky on Windows due to PR24554, so add some
# retries. We don't do this on otther platforms because it's slower.
if platform.system() == 'Windows':
  config.test_retry_attempts = 2

# No throttling on non-Darwin platforms.
lit_config.parallelism_groups['shadow-memory'] = None

if platform.system() == 'Darwin':
  ios_device = config.apple_platform != 'osx' and not config.apple_platform.endswith('sim')
  # Force sequential execution when running tests on iOS devices.
  if ios_device:
    lit_config.warning('Forcing sequential execution for iOS device tests')
    lit_config.parallelism_groups['ios-device'] = 1
    config.parallelism_group = 'ios-device'

  # Only run up to 3 processes that require shadow memory simultaneously on
  # 64-bit Darwin. Using more scales badly and hogs the system due to
  # inefficient handling of large mmap'd regions (terabytes) by the kernel.
  else:
    lit_config.warning('Throttling sanitizer tests that require shadow memory on Darwin')
    lit_config.parallelism_groups['shadow-memory'] = 3

# Multiple substitutions are necessary to support multiple shared objects used
# at once.
# Note that substitutions with numbers have to be defined first to avoid
# being subsumed by substitutions with smaller postfix.
for postfix in ["2", "1", ""]:
  if config.host_os == 'Darwin':
    config.substitutions.append( ("%ld_flags_rpath_exe" + postfix, '-Wl,-rpath,@executable_path/ %dynamiclib' + postfix) )
    config.substitutions.append( ("%ld_flags_rpath_so" + postfix, '-install_name @rpath/`basename %dynamiclib{}`'.format(postfix)) )
  elif config.host_os in ('FreeBSD', 'NetBSD', 'OpenBSD'):
    config.substitutions.append( ("%ld_flags_rpath_exe" + postfix, "-Wl,-z,origin -Wl,-rpath,\$ORIGIN -L%T -l%xdynamiclib_namespec" + postfix) )
    config.substitutions.append( ("%ld_flags_rpath_so" + postfix, '') )
  elif config.host_os == 'Linux':
    config.substitutions.append( ("%ld_flags_rpath_exe" + postfix, "-Wl,-rpath,\$ORIGIN -L%T -l%xdynamiclib_namespec" + postfix) )
    config.substitutions.append( ("%ld_flags_rpath_so" + postfix, '') )
  elif config.host_os == 'SunOS':
    config.substitutions.append( ("%ld_flags_rpath_exe" + postfix, "-Wl,-R\$ORIGIN -L%T -l%xdynamiclib_namespec" + postfix) )
    config.substitutions.append( ("%ld_flags_rpath_so" + postfix, '') )

  # Must be defined after the substitutions that use %dynamiclib.
  config.substitutions.append( ("%dynamiclib" + postfix, '%T/%xdynamiclib_filename' + postfix) )
  config.substitutions.append( ("%xdynamiclib_filename" + postfix, 'lib%xdynamiclib_namespec{}.so'.format(postfix)) )
  config.substitutions.append( ("%xdynamiclib_namespec", '%basename_t.dynamic') )

# Provide a substitution that can be used to tell Clang to use a static libstdc++.
# The substitution expands to nothing on non Linux platforms.
# FIXME: This should check the target OS, not the host OS.
if config.host_os == 'Linux':
  config.substitutions.append( ("%linux_static_libstdcplusplus", "-stdlib=libstdc++ -static-libstdc++") )
else:
  config.substitutions.append( ("%linux_static_libstdcplusplus", "") )

config.default_sanitizer_opts = []
if config.host_os == 'Darwin':
  # On Darwin, we default to `abort_on_error=1`, which would make tests run
  # much slower. Let's override this and run lit tests with 'abort_on_error=0'.
  config.default_sanitizer_opts += ['abort_on_error=0']
  config.default_sanitizer_opts += ['log_to_syslog=0']
  if lit.util.which('log'):
    # Querying the log can only done by a privileged user so
    # so check if we can query the log.
    exit_code = -1
    with open('/dev/null', 'r') as f:
      # Run a `log show` command the should finish fairly quickly and produce very little output.
      exit_code = subprocess.call(['log', 'show', '--last', '1m', '--predicate', '1 == 0'], stdout=f, stderr=f)
    if exit_code == 0:
      config.available_features.add('darwin_log_cmd')
    else:
      lit_config.warning('log command found but cannot queried')
  else:
    lit_config.warning('log command not found. Some tests will be skipped.')
elif config.android:
  config.default_sanitizer_opts += ['abort_on_error=0']

# Allow tests to use REQUIRES=stable-runtime.  For use when you cannot use XFAIL
# because the test hangs or fails on one configuration and not the other.
if config.android or (config.target_arch not in ['arm', 'armhf', 'aarch64']):
  config.available_features.add('stable-runtime')

if config.asan_shadow_scale:
  config.available_features.add("shadow-scale-%s" % config.asan_shadow_scale)
else:
  config.available_features.add("shadow-scale-3")

if config.memprof_shadow_scale:
  config.available_features.add("memprof-shadow-scale-%s" % config.memprof_shadow_scale)
else:
  config.available_features.add("memprof-shadow-scale-3")

if config.expensive_checks:
  config.available_features.add("expensive_checks")

# Propagate the LLD/LTO into the clang config option, so nothing else is needed.
run_wrapper = []
target_cflags = [getattr(config, 'target_cflags', None)]
extra_cflags = []

if config.use_lto and config.lto_supported:
  extra_cflags += config.lto_flags
elif config.use_lto and (not config.lto_supported):
  config.unsupported = True

if config.use_lld and config.has_lld and not config.use_lto:
  extra_cflags += ["-fuse-ld=lld"]
elif config.use_lld and (not config.has_lld):
  config.unsupported = True

# Append any extra flags passed in lit_config
append_target_cflags = lit_config.params.get('append_target_cflags', None)
if append_target_cflags:
  lit_config.note('Appending to extra_cflags: "{}"'.format(append_target_cflags))
  extra_cflags += [append_target_cflags]

config.clang = " " + " ".join(run_wrapper + [config.compile_wrapper, config.clang]) + " "
config.target_cflags = " " + " ".join(target_cflags + extra_cflags) + " "

if config.host_os == 'Darwin':
  config.substitutions.append((
    "%get_pid_from_output",
    "{} {}/get_pid_from_output.py".format(
      sh_quote(config.python_executable),
      sh_quote(get_ios_commands_dir())
    ))
  )
  config.substitutions.append(
    ("%print_crashreport_for_pid",
    "{} {}/print_crashreport_for_pid.py".format(
      sh_quote(config.python_executable),
      sh_quote(get_ios_commands_dir())
    ))
  )

# It is not realistically possible to account for all options that could
# possibly be present in system and user configuration files, so disable
# default configs for the test runs. In particular, anything hardening
# related is likely to cause issues with sanitizer tests, because it may
# preempt something we're looking to trap (e.g. _FORTIFY_SOURCE vs our ASAN).
config.environment["CLANG_NO_DEFAULT_CONFIG"] = "1"
