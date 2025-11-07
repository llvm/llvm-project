import argparse
import os
import subprocess
import sys
import traceback
import util
import traceback
import subprocess
from contextlib import contextmanager

def relative_if_possible(path, relative_to):
        path = os.path.normpath(path)
        try:
            result = os.path.relpath(path, start=relative_to)
            return result if result else path
        except Exception:
            return  path


def common_init(jobs=None):
    # For information
    for k,v in os.environ.items():
        print(f"{k}={v}")

    jobs_default = None
    if jobsenv := os.environ.get('BUILDBOT_JOBS'):
                jobs_default = int(jobsenv)
    if not jobs_default:
            jobs_default = jobs

    parser = argparse.ArgumentParser(allow_abbrev=True, description="Run the buildbot builder configuration; when executed without arguments, builds the llvm-project checkout this file is in, using cwd to write all files")
    parser.add_argument('--jobs', '-j', default=jobs_default, help='Override the number fo default jobs')
    parser.add_argument('--clean', type=bool, default=os.environ.get('BUILDBOT_CLEAN'), help='Whether to delete source-, build-, and install-dirs (i.e. remove any files leftover from a previous run, if any) before running')
    parser.add_argument('--clobber', type=bool, default=os.environ.get('BUILDBOT_CLOBBER'), help='Whether to delete build- and install-dirs before running')
    parser.add_argument('--workdir', help="Use this dir as workdir (default: current working directory)" )
    return parser


@contextmanager
def step(step_name, halt_on_fail=False):
    sys.stderr.flush()
    sys.stdout.flush()
    util.report('@@@BUILD_STEP {}@@@'.format(step_name))
    if halt_on_fail:
        util.report('@@@HALT_ON_FAILURE@@@')
    try:
        yield
    except Exception as e:
        if isinstance(e, subprocess.CalledProcessError):
            util.report(
                '{} exited with return code {}.'.format(e.cmd, e.returncode)
            )
        else:
            util.report('The build step threw an exception...')
            traceback.print_exc()

        util.report('@@@STEP_FAILURE@@@')
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
    util.report_run_cmd(cmd, **kwargs)


def run_ninja(args, targets, ccache_stats=False, **kwargs):
    cmd = ['ninja', *targets]
    if args.jobs:
        args .append(f'-j{args.jobs}')
    if ccache_stats:
            util.report_run_cmd(['ccache', '-z'])
    util.report_run_cmd(cmd, **kwargs)
    if ccache_stats:
            util.report_run_cmd(['ccache', '-sv'])



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
          util.clean_dir(d)

