import distutils.sysconfig
import os
import platform
import re
import sys


def get_python_relative_libdir():
    """Returns the appropropriate python libdir relative to the build directory.

    @param exe_path the path to the lldb executable

    @return the python path that needs to be added to sys.path (PYTHONPATH)
    in order to find the lldb python module.
    """
    if platform.system() != 'Linux':
        return None

    # We currently have a bug in lldb -P that does not account for
    # architecture variants in python paths for
    # architecture-specific modules.  Handle the lookup here.
    # When that bug is fixed, we should just ask lldb for the
    # right answer always.
    arch_specific_libdir = distutils.sysconfig.get_python_lib(True, False)
    split_libdir = arch_specific_libdir.split(os.sep)
    lib_re = re.compile(r"^lib.*$")

    for i in range(len(split_libdir)):
        match = lib_re.match(split_libdir[i])
        if match is not None:
            # We'll call this the relative root of the lib dir.
            # Things like RHEL will have an arch-specific python
            # lib dir, which isn't 'lib' on x86_64.
            return os.sep.join(split_libdir[i:])
    # Didn't resolve it.
    return None

if __name__ == '__main__':
    lib_dir = get_python_relative_libdir()
    if lib_dir is not None:
        sys.stdout.write(lib_dir)
        sys.exit(0)
    else:
        sys.exit(1)
