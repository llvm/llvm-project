import argparse
import filecmp
import tempfile
import os
import subprocess
import sys
import traceback
import pathlib
import subprocess
import errno
import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager






def cmake_pjoin(*args):
    """
    Join paths like safe_pjoin, but replace backslashes with forward
    slashes on platforms where they are path separators. This prevents
    CMake from choking when trying to decode what it thinks are escape
    sequences in filenames.
    """
    result = safe_pjoin(*args)
    if os.sep == '\\':
        return result.replace('\\', '/')
    else:
        return result


def report(msg):
    sys.stderr.write(msg + '\n')
    sys.stderr.flush()


def report_run_cmd(cmd, shell=False, *args, **kwargs):
    """
    Print a command, then executes it using subprocess.check_call.
    """
    report('Running: %s' % ((cmd if shell else shquote_cmd(cmd)),))
    sys.stderr.flush()
    subprocess.check_call(cmd, shell=shell, *args, **kwargs)


def mkdirp(path):
    """Create directory path if it does not already exist."""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def rmtree(path):
    """
    Remove directory path and all its subdirectories. This differs from
    shutil.rmtree() in that it tries to adjust permissions so that deletion
    will succeed.
    """
    # Some files will not be deletable, so we set permissions that allow
    # deletion before we try deleting files.
    for root, dirs, files in os.walk(path):
        os.chmod(root, 0o755)
        for f in files:
            p = os.path.join(root, f)
            os.chmod(p, 0o644)
            os.unlink(p)
    # At this point, we should have a tree of deletable directories.
    shutil.rmtree(path)


def safe_pjoin(dirname, *args):
    """
    Join path components with os.path.join, skipping the first component
    if it is None.
    """
    if dirname is None:
        return os.path.join(*args)
    else:
        return os.path.join(dirname, *args)


def _shquote_impl(txt, escaped_chars, quoted_chars):
    quoted = re.sub(escaped_chars, r'\\\1', txt)
    if len(quoted) == len(txt) and not quoted_chars.search(txt):
        return txt
    else:
        return '"' + quoted + '"'


_SHQUOTE_POSIX_ESCAPEDCHARS = re.compile(r'(["`$\\])')
_SHQUOTE_POSIX_QUOTEDCHARS = re.compile('[|&;<>()\' \t\n]')


def shquote_posix(txt):
    """Return txt, appropriately quoted for POSIX shells."""
    return _shquote_impl(
        txt, _SHQUOTE_POSIX_ESCAPEDCHARS, _SHQUOTE_POSIX_QUOTEDCHARS)


_SHQUOTE_WINDOWS_ESCAPEDCHARS = re.compile(r'(["\\])')
_SHQUOTE_WINDOWS_QUOTEDCHARS = re.compile('[ \t\n]')


def shquote_windows(txt):
    """Return txt, appropriately quoted for Windows's cmd.exe."""
    return _shquote_impl(
        txt.replace('%', '%%'),
        _SHQUOTE_WINDOWS_ESCAPEDCHARS, _SHQUOTE_WINDOWS_QUOTEDCHARS)


def shquote(txt):
    """Return txt, appropriately quoted for use in a shell command."""
    if os.name in set(('nt', 'os2', 'ce')):
        return shquote_windows(txt)
    else:
        return shquote_posix(txt)


def shquote_cmd(cmd):
    """Convert a list of shell arguments to an appropriately quoted string."""
    return ' '.join(map(shquote, cmd))



def clean_dir(path):
    """
    Removes directory at path (and all its subdirectories) if it exists,
    and creates an empty directory in its place.
    """
    try:
        rmtree(path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise
    mkdirp(path)


def cmake_pjoin(*args):
    """
    Join paths like safe_pjoin, but replace backslashes with forward
    slashes on platforms where they are path separators. This prevents
    CMake from choking when trying to decode what it thinks are escape
    sequences in filenames.
    """
    result = safe_pjoin(*args)
    if os.sep == '\\':
        return result.replace('\\', '/')
    else:
        return result


def report(msg):
    sys.stderr.write(msg + '\n')
    sys.stderr.flush()


def report_run_cmd(cmd, shell=False, *args, **kwargs):
    """
    Print a command, then executes it using subprocess.check_call.
    """
    report('Running: %s' % ((cmd if shell else shquote_cmd(cmd)),))
    sys.stderr.flush()
    subprocess.check_call([str(c) for c in  cmd], shell=shell, *args, **kwargs)


def mkdirp(path):
    """Create directory path if it does not already exist."""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def rmtree(path):
    """
    Remove directory path and all its subdirectories. This differs from
    shutil.rmtree() in that it tries to adjust permissions so that deletion
    will succeed.
    """
    # Some files will not be deletable, so we set permissions that allow
    # deletion before we try deleting files.
    for root, dirs, files in os.walk(path):
        os.chmod(root, 0o755)
        for f in files:
            p = os.path.join(root, f)
            os.chmod(p, 0o644)
            os.unlink(p)
    # At this point, we should have a tree of deletable directories.
    shutil.rmtree(path)


def safe_pjoin(dirname, *args):
    """
    Join path components with os.path.join, skipping the first component
    if it is None.
    """
    if dirname is None:
        return os.path.join(*args)
    else:
        return os.path.join(dirname, *args)


def _shquote_impl(txt, escaped_chars, quoted_chars):
    quoted = re.sub(escaped_chars, r'\\\1', txt)
    if len(quoted) == len(txt) and not quoted_chars.search(txt):
        return txt
    else:
        return '"' + quoted + '"'


_SHQUOTE_POSIX_ESCAPEDCHARS = re.compile(r'(["`$\\])')
_SHQUOTE_POSIX_QUOTEDCHARS = re.compile('[|&;<>()\' \t\n]')


def shquote_posix(txt):
    """Return txt, appropriately quoted for POSIX shells."""
    return _shquote_impl(
        txt, _SHQUOTE_POSIX_ESCAPEDCHARS, _SHQUOTE_POSIX_QUOTEDCHARS)


_SHQUOTE_WINDOWS_ESCAPEDCHARS = re.compile(r'(["\\])')
_SHQUOTE_WINDOWS_QUOTEDCHARS = re.compile('[ \t\n]')


def shquote_windows(txt):
    """Return txt, appropriately quoted for Windows's cmd.exe."""
    return _shquote_impl(
        txt.replace('%', '%%'),
        _SHQUOTE_WINDOWS_ESCAPEDCHARS, _SHQUOTE_WINDOWS_QUOTEDCHARS)


def shquote(txt):
    """Return txt, appropriately quoted for use in a shell command."""
    if os.name in set(('nt', 'os2', 'ce')):
        return shquote_windows(txt)
    else:
        return shquote_posix(txt)


def shquote_cmd(cmd):
    """Convert a list of shell arguments to an appropriately quoted string."""
    return ' '.join(map(shquote, cmd))


def relative_if_possible(path, relative_to):
        path = os.path.normpath(path)
        try:
            result = os.path.relpath(path, start=relative_to)
            return result if result else path
        except Exception:
            return  path




def first_true(*args):
    for a in args:
        if a:
            return a
    return None


def first_nonnull(*args):
    for a in args:
        if a:
            return a
    return None


class Worker:
    def __init__(self,args,clean,clobber,workdir,jobs,cachefile,llvmsrcroot):
        self.args=args
        self.clean =clean
        self.clobber=clobber
        self.workdir=workdir
        self.jobs=jobs
        self.cachefile =cachefile
        self.llvmsrcroot=llvmsrcroot

    def in_llvmsrc(self, path):
        return os.path.join(self.llvmsrcroot, path)

    def in_workdir(self, path):
        return os.path.join(self.workdir, path)

    @contextmanager
    def step(self, step_name, halt_on_fail=False):
        with step(step_name,halt_on_fail=halt_on_fail) as s:
            yield s

    def run_cmake(self, cmakeargs):
        report_run_cmd(['cmake', *cmakeargs])

    def run_ninja(args, targets, ccache_stats=False, **kwargs):
        cmd = ['ninja', *targets]
        if args.jobs:
            args .append(f'-j{args.jobs}')
        if ccache_stats:
                report_run_cmd(['ccache', '-z'])
        report_run_cmd(cmd, **kwargs)
        if ccache_stats:
                report_run_cmd(['ccache', '-sv'])

    def checkout (self,giturl, sourcepath):
            return checkout(giturl, sourcepath)

 




@contextmanager
def run(scriptname, llvmsrcroot, parser=None ,clobberpaths=[], workerjobs=None):
    for k,v in os.environ.items():
        print(f"{k}={v}")

    os.environ['NINJA_STATUS'] = "[%p/%es :: %u->%r->%f (of %t)] "

    jobs_default = first_true(  os.environ.get('BUILDBOT_JOBS') , workerjobs )



    stem = pathlib.Path(scriptname).stem


    parser =  parser or argparse.ArgumentParser(allow_abbrev=True, description="Run the buildbot builder configuration; when executed without arguments, builds the llvm-project checkout this file is in, using cwd to write all files")
    parser.add_argument('--jobs', '-j', default=jobs_default, help='Override the number of default jobs')
    parser.add_argument('--clean', type=bool, default=os.environ.get('BUILDBOT_CLEAN') , help='Whether to delete source-, build-, and install-dirs (i.e. remove any files leftover from a previous run, if any) before running')
    parser.add_argument('--clobber', type=bool, default=os.environ.get('BUILDBOT_CLOBBER') or os.environ.get('BUILDBOT_CLEAN_OBJ'), help='Whether to delete build- and install-dirs before running')
    parser.add_argument('--workdir', default=f'{stem}.workdir', help="Use this dir as workdir")
    parser.add_argument('--cachefile', default=relative_if_possible(pathlib.Path(scriptname).with_suffix('.cmake'), llvmsrcroot), help='CMake cache seed')
    args = parser.parse_args()

    print("cwd:",os.getcwd())

    workdir =  args.workdir
    clobber = args.clobber
    cachefile = os.path.join(llvmsrcroot, args.cachefile)
    oldcwd = os.getcwd()


    prevcachepath = os.path.join(workdir, 'prevcache.cmake')
    if cachefile and os.path.exists(prevcachepath):
        # Force clobber if cache file has changed; a new cachefile does not override entries already present in CMakeCache.txt
        if   not filecmp.cmp( os.path.join(llvmsrcroot, args.cachefile), prevcachepath, shallow=False):
            clobber = True
        #shutil.copyfile(cachefile)

    w = Worker(args, clean=args.clean , clobber=clobber, workdir=workdir, jobs=args.jobs, cachefile=cachefile, llvmsrcroot=llvmsrcroot)

    #if workdir:
    if args.clean:
            with w.step(f'clean'):
                clean_dir(workdir)
    elif clobber:
       with w.step(f'clobber'):
            for d in clobberpaths:
                    clean_dir(os.path.join(workdir, d))
            os.path.unlink(prevcachepath)

    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    # Remember used cachefile in case it changes
    if cachefile:
        shutil.copy2(os.path.join(oldcwd,llvmsrcroot, args.cachefile), os.path.join(oldcwd, prevcachepath ))

    yield w
    #else:
    #    with tempfile.TemporaryDirectory(prefix = stem) as tmpdir:
    #                os.chdir(tmpdir)
    #                yield Worker(args)






@contextmanager
def step(step_name, halt_on_fail=False):
    sys.stderr.flush()
    sys.stdout.flush()
    report(f'@@@BUILD_STEP {step_name}@@@')
    if halt_on_fail:
        report('@@@HALT_ON_FAILURE@@@')
    try:
        yield
    except Exception as e:
        if isinstance(e, subprocess.CalledProcessError):
            report(f'{e.cmd} exited with return code {e.returncode}.' )
            report('@@@STEP_FAILURE@@@')
        else:
            traceback.print_exc()
            report('@@@STEP_EXCEPTION@@@')
        if halt_on_fail:
            exit(1)


def get_steps(makefile):
    try:
        make_cmd = build_make_cmd(makefile, 'get-steps')
        raw_steps = capture_cmd_stdout(make_cmd)
        return raw_steps.decode('utf-8').split('\n')[:-1]
    except:
        return []


def build_make_cmd(makefile, target, make_vars={}):
    make_cmd = ['make', '-f', makefile]
    if not target is None:
        make_cmd.append(target)
    for k,v in make_vars.items():
        make_cmd += ["{}={}".format(k, v)]
    return make_cmd


def capture_cmd_stdout(cmd, **kwargs):
    return subprocess.run(cmd, shell=False, check=True, stdout=subprocess.PIPE, **kwargs).stdout


def run_command(cmd, **kwargs):
    report_run_cmd(cmd, **kwargs)





def checkout(giturl, sourcepath):
    if not os.path.exists(sourcepath):
        run_command(['git', 'clone', giturl, sourcepath])

    # Reset repository state no matter what there was before
    run_command(['git', '-C', sourcepath, 'stash', '--all'])
    run_command(['git', '-C', sourcepath, 'stash', 'clear'])

    # Fetch and checkout the newest
    run_command(['git', '-C', sourcepath, 'fetch', 'origin'])
    run_command(['git', '-C', sourcepath, 'checkout', 'origin/main', '--detach'])


def clean_on_request(args, always=[],on_clobber=[],on_clean=[]):
    cleanset = always
    if args.clobber or args.clean:
        # Clean implies clobber
        cleanset += on_clobber
    if  args.clean:
        cleanset += on_clean

    for d in cleanset:
        with step(f'delete-{os.path.basename(d)}'):
          clean_dir(d)

