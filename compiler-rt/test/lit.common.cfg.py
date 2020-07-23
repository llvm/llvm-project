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

# Setup test format.
config.test_format = lit.formats.ShTest(execute_external)
if execute_external:
  config.available_features.add('shell')

compiler_id = getattr(config, 'compiler_id', None)
if compiler_id == "Clang":
  if platform.system() != 'Windows':
    config.cxx_mode_flags = ["--driver-mode=g++"]
  else:
    config.cxx_mode_flags = []
  # We assume that sanitizers should provide good enough error
  # reports and stack traces even with minimal debug info.
  config.debug_info_flags = ["-gline-tables-only"]
  if platform.system() == 'Windows':
    # On Windows, use CodeView with column info instead of DWARF. Both VS and
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

# If needed, add cflag for shadow scale.
if config.asan_shadow_scale != '':
  config.target_cflags += " -mllvm -asan-mapping-scale=" + config.asan_shadow_scale

# BFD linker in 64-bit android toolchains fails to find libc++_shared.so, which
# is a transitive shared library dependency (via asan runtime).
if config.android:
  # Prepend the flag so that it can be overridden.
  config.target_cflags = "-pie -fuse-ld=gold " + config.target_cflags
  if config.android_ndk_version < 19:
    # With a new compiler and NDK < r19 this flag ends up meaning "link against
    # libc++", but NDK r19 makes this mean "link against the stub libstdc++ that
    # just contains a handful of ABI functions", which makes most C++ code fail
    # to link. In r19 and later we just use the default which is libc++.
    config.cxx_mode_flags.append('-stdlib=libstdc++')

config.environment = dict(os.environ)

# Clear some environment variables that might affect Clang.
possibly_dangerous_env_vars = ['ASAN_OPTIONS', 'DFSAN_OPTIONS', 'LSAN_OPTIONS',
                               'MSAN_OPTIONS', 'UBSAN_OPTIONS',
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
# Clang/Win32 may refer to %INCLUDE%. vsvarsall.bat sets it.
if platform.system() != 'Windows':
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
if platform.system() == 'Windows' and '-win' in config.target_triple:
  config.environment['LIB'] = os.environ['LIB']

config.available_features.add(config.host_os.lower())

if re.match(r'^x86_64.*-linux', config.target_triple):
  config.available_features.add("x86_64-linux")

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

# Allow tests to be executed on a simulator or remotely.
if config.emulator:
  config.substitutions.append( ('%run', config.emulator) )
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

  ios_commands_dir = os.path.join(config.compiler_rt_src_root, "test", "sanitizer_common", "ios_commands")

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
  # TODO: Implement `%device_rm` to perform removal of files on a device.  For
  # now just make it a no-op.
  lit_config.warning('%device_rm is not implemented')
  config.substitutions.append( ('%device_rm', 'echo ') )
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

if config.has_lld:
  config.available_features.add('lld-available')

if config.use_lld:
  config.available_features.add('lld')

if config.can_symbolize:
  config.available_features.add('can-symbolize')

if config.gwp_asan:
  config.available_features.add('gwp_asan')

lit.util.usePlatformSdkOnDarwin(config, lit_config)

# Maps a lit substitution name for the minimum target OS flag
# to the macOS version that first contained the relevant feature.
darwin_min_deployment_target_substitutions = {
  '%macos_min_target_10_11': '10.11',
  # rdar://problem/22207160
  '%darwin_min_target_with_full_runtime_arc_support': '10.11',
  '%darwin_min_target_with_tls_support': '10.12',
}

if config.host_os == 'Darwin':
  def get_apple_platform_version_aligned_with(macos_version, apple_platform):
    """
      Given a macOS version (`macos_version`) returns the corresponding version for
      the specified Apple platform if it exists.

      `macos_version` - The macOS version as a string.
      `apple_platform` - The Apple platform name as a string.

      Returns the corresponding version as a string if it exists, otherwise
      `None` is returned.
    """
    m = re.match(r'^10\.(?P<min>\d+)(\.(?P<patch>\d+))?$', macos_version)
    if not m:
      raise Exception('Could not parse macOS version: "{}"'.format(macos_version))
    ver_min = int(m.group('min'))
    ver_patch = m.group('patch')
    if ver_patch:
      ver_patch = int(ver_patch)
    else:
      ver_patch = 0
    result_str = ''
    if apple_platform == 'osx':
      # Drop patch for now.
      result_str = '10.{}'.format(ver_min)
    elif apple_platform.startswith('ios') or apple_platform.startswith('tvos'):
      result_maj = ver_min - 2
      if result_maj < 1:
        return None
      result_str = '{}.{}'.format(result_maj, ver_patch)
    elif apple_platform.startswith('watch'):
      result_maj = ver_min - 9
      if result_maj < 1:
        return None
      result_str = '{}.{}'.format(result_maj, ver_patch)
    else:
      raise Exception('Unsuported apple platform "{}"'.format(apple_platform))
    return result_str

  osx_version = (10, 0, 0)
  try:
    osx_version = subprocess.check_output(["sw_vers", "-productVersion"],
                                          universal_newlines=True)
    osx_version = tuple(int(x) for x in osx_version.split('.'))
    if len(osx_version) == 2: osx_version = (osx_version[0], osx_version[1], 0)
    if osx_version >= (10, 11):
      config.available_features.add('osx-autointerception')
      config.available_features.add('osx-ld64-live_support')
    else:
      # The ASAN initialization-bug.cpp test should XFAIL on OS X systems
      # older than El Capitan. By marking the test as being unsupported with
      # this "feature", we can pass the test on newer OS X versions and other
      # platforms.
      config.available_features.add('osx-no-ld64-live_support')
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

  def get_apple_min_deploy_target_flag_aligned_with_osx(version):
    min_os_aligned_with_osx_v = get_apple_platform_version_aligned_with(version, config.apple_platform)
    min_os_aligned_with_osx_v_flag = ''
    if min_os_aligned_with_osx_v:
      min_os_aligned_with_osx_v_flag = '{flag}={version}'.format(
        flag=config.apple_platform_min_deployment_target_flag,
        version=min_os_aligned_with_osx_v)
    else:
      lit_config.warning('Could not find a version of {} that corresponds with macOS {}'.format(
        config.apple_platform,
        version))
    return min_os_aligned_with_osx_v_flag

  for substitution, osx_version in darwin_min_deployment_target_substitutions.items():
    config.substitutions.append( (substitution, get_apple_min_deploy_target_flag_aligned_with_osx(osx_version)) )

  # 32-bit iOS simulator is deprecated and removed in latest Xcode.
  if config.apple_platform == "iossim":
    if config.target_arch == "i386":
      config.unsupported = True
else:
  for substitution in darwin_min_deployment_target_substitutions.keys():
    config.substitutions.append( (substitution, "") )

if config.android:
  env = os.environ.copy()
  if config.android_serial:
    env['ANDROID_SERIAL'] = config.android_serial
    config.environment['ANDROID_SERIAL'] = config.android_serial

  adb = os.environ.get('ADB', 'adb')
  try:
    android_api_level_str = subprocess.check_output([adb, "shell", "getprop", "ro.build.version.sdk"], env=env).rstrip()
  except (subprocess.CalledProcessError, OSError):
    lit_config.fatal("Failed to read ro.build.version.sdk (using '%s' as adb)" % adb)
  try:
    android_api_level = int(android_api_level_str)
  except ValueError:
    lit_config.fatal("Failed to read ro.build.version.sdk (using '%s' as adb): got '%s'" % (adb, android_api_level_str))
  if android_api_level >= 26:
    config.available_features.add('android-26')
  if android_api_level >= 28:
    config.available_features.add('android-28')

  # Prepare the device.
  android_tmpdir = '/data/local/tmp/Output'
  subprocess.check_call([adb, "shell", "mkdir", "-p", android_tmpdir], env=env)
  for file in config.android_files_to_push:
    subprocess.check_call([adb, "push", file, android_tmpdir], env=env)

if config.host_os == 'Linux':
  # detect whether we are using glibc, and which version
  # NB: 'ldd' is just one of the tools commonly installed as part of glibc
  ldd_ver_cmd = subprocess.Popen(['ldd', '--version'],
                                 stdout=subprocess.PIPE,
                                 env={'LANG': 'C'})
  sout, _ = ldd_ver_cmd.communicate()
  ver_line = sout.splitlines()[0]
  if ver_line.startswith(b"ldd "):
    from distutils.version import LooseVersion
    ver = LooseVersion(ver_line.split()[-1].decode())
    # 2.27 introduced some incompatibilities
    if ver >= LooseVersion("2.27"):
      config.available_features.add("glibc-2.27")

sancovcc_path = os.path.join(config.llvm_tools_dir, "sancov")
if os.path.exists(sancovcc_path):
  config.available_features.add("has_sancovcc")
  config.substitutions.append( ("%sancovcc ", sancovcc_path) )

def is_darwin_lto_supported():
  return os.path.exists(os.path.join(config.llvm_shlib_dir, 'libLTO.dylib'))

def is_linux_lto_supported():
  if config.use_lld:
    return True

  if not os.path.exists(os.path.join(config.llvm_shlib_dir, 'LLVMgold.so')):
    return False

  ld_cmd = subprocess.Popen([config.gold_executable, '--help'], stdout = subprocess.PIPE, env={'LANG': 'C'})
  ld_out = ld_cmd.stdout.read().decode()
  ld_cmd.wait()

  if not '-plugin' in ld_out:
    return False

  return True

def is_windows_lto_supported():
  return os.path.exists(os.path.join(config.llvm_tools_dir, 'lld-link.exe'))

if config.host_os == 'Darwin' and is_darwin_lto_supported():
  config.lto_supported = True
  config.lto_launch = ["env", "DYLD_LIBRARY_PATH=" + config.llvm_shlib_dir]
  config.lto_flags = []
elif config.host_os in ['Linux', 'FreeBSD', 'NetBSD'] and is_linux_lto_supported():
  config.lto_supported = True
  config.lto_launch = []
  if config.use_lld:
    config.lto_flags = ["-fuse-ld=lld"]
  else:
    config.lto_flags = ["-fuse-ld=gold"]
elif config.host_os == 'Windows' and is_windows_lto_supported():
  config.lto_supported = True
  config.lto_launch = []
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
  if config.use_newpm:
    config.lto_flags += ["-fexperimental-new-pass-manager"]

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
  elif config.target_arch in ['x86_64', 'x86_64h']:
    lit_config.warning('Throttling sanitizer tests that require shadow memory on Darwin 64bit')
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

# Provide a substituion that can be used to tell Clang to use a static libstdc++.
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

if config.expensive_checks:
  config.available_features.add("expensive_checks")

# Propagate the LLD/LTO into the clang config option, so nothing else is needed.
run_wrapper = []
target_cflags = [getattr(config, 'target_cflags', None)]
extra_cflags = []

if config.use_lto and config.lto_supported:
  run_wrapper += config.lto_launch
  extra_cflags += config.lto_flags
elif config.use_lto and (not config.lto_supported):
  config.unsupported = True

if config.use_lld and config.has_lld and not config.use_lto:
  extra_cflags += ["-fuse-ld=lld"]
elif config.use_lld and (not config.has_lld):
  config.unsupported = True

config.clang = " " + " ".join(run_wrapper + [config.compile_wrapper, config.clang]) + " "
config.target_cflags = " " + " ".join(target_cflags + extra_cflags) + " "
