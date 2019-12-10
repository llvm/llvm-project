#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

import platform
import os
import posixpath
import ntpath

from libcxx.test import tracing
from libcxx.util import executeCommand

class Executor(object):
    def __init__(self):
        self.target_info = None

    def run(self, exe_path, cmd, local_cwd, file_deps=None, env=None):
        """Execute a command.
            Be very careful not to change shared state in this function.
            Executor objects are shared between python processes in `lit -jN`.
        Args:
            exe_path: str:    Local path to the executable to be run
            cmd: [str]:       subprocess.call style command
            local_cwd: str:   Local path to the working directory
            file_deps: [str]: Files required by the test
            env: {str: str}:  Environment variables to execute under
        Returns:
            cmd, out, err, exitCode
        """
        raise NotImplementedError

    def merge_environments(self, current_env, updated_env):
        """Merges two execution environments.

        If both environments contain the PATH variables, they are also merged
        using the proper separator.
        """
        result_env = dict(current_env)
        for k, v in updated_env.items():
            if k == 'PATH' and self.target_info:
                self.target_info.add_path(result_env, v)
            else:
                result_env[k] = v
        return result_env


class LocalExecutor(Executor):
    def __init__(self):
        super(LocalExecutor, self).__init__()
        self.is_windows = platform.system() == 'Windows'

    def run(self, exe_path, cmd=None, work_dir='.', file_deps=None, env=None):
        cmd = cmd or [exe_path]
        if work_dir == '.':
            work_dir = os.getcwd()

        if env:
            env = self.merge_environments(os.environ, env)

        out, err, rc = executeCommand(cmd, cwd=work_dir, env=env)
        return (cmd, out, err, rc)


class PrefixExecutor(Executor):
    """Prefix an executor with some other command wrapper.

    Most useful for setting ulimits on commands, or running an emulator like
    qemu and valgrind.
    """
    def __init__(self, commandPrefix, chain):
        super(PrefixExecutor, self).__init__()

        self.commandPrefix = commandPrefix
        self.chain = chain

    def run(self, exe_path, cmd=None, work_dir='.', file_deps=None, env=None):
        cmd = cmd or [exe_path]
        return self.chain.run(exe_path, self.commandPrefix + cmd, work_dir,
                              file_deps, env=env)


class PostfixExecutor(Executor):
    """Postfix an executor with some args."""
    def __init__(self, commandPostfix, chain):
        super(PostfixExecutor, self).__init__()

        self.commandPostfix = commandPostfix
        self.chain = chain

    def run(self, exe_path, cmd=None, work_dir='.', file_deps=None, env=None):
        cmd = cmd or [exe_path]
        return self.chain.run(cmd + self.commandPostfix, work_dir, file_deps,
                              env=env)



class TimeoutExecutor(PrefixExecutor):
    """Execute another action under a timeout.

    Deprecated. http://reviews.llvm.org/D6584 adds timeouts to LIT.
    """
    def __init__(self, duration, chain):
        super(TimeoutExecutor, self).__init__(
            ['timeout', duration], chain)


class RemoteExecutor(Executor):
    def __init__(self):
        super(RemoteExecutor, self).__init__()
        self.local_run = executeCommand

    def remote_temp_dir(self):
        return self._remote_temp(True)

    def remote_temp_file(self):
        return self._remote_temp(False)

    def _remote_temp(self, is_dir):
        raise NotImplementedError()

    def copy_in(self, local_srcs, remote_dsts):
        # This could be wrapped up in a tar->scp->untar for performance
        # if there are lots of files to be copied/moved
        for src, dst in zip(local_srcs, remote_dsts):
            self._copy_in_file(src, dst)

    def _copy_in_file(self, src, dst):
        raise NotImplementedError()

    def delete_remote(self, remote):
        try:
            self._execute_command_remote(['rm', '-rf', remote])
        except OSError:
            # TODO: Log failure to delete?
            pass

    def run(self, exe_path, cmd=None, work_dir='.', file_deps=None, env=None):
        target_exe_path = None
        target_cwd = None
        try:
            target_cwd = self.remote_temp_dir()
            executable_name = 'libcxx_test.exe'
            if self.target_info.is_windows():
                target_exe_path = ntpath.join(target_cwd, executable_name)
            else:
                target_exe_path = posixpath.join(target_cwd, executable_name)

            if cmd:
                # Replace exe_path with target_exe_path.
                cmd = [c if c != exe_path else target_exe_path for c in cmd]
            else:
                cmd = [target_exe_path]

            srcs = [exe_path]
            dsts = [target_exe_path]
            if file_deps is not None:
                dev_paths = [os.path.join(target_cwd, os.path.basename(f))
                             for f in file_deps]
                srcs.extend(file_deps)
                dsts.extend(dev_paths)
            self.copy_in(srcs, dsts)

            # When testing executables that were cross-compiled on Windows for
            # Linux, we may need to explicitly set the execution permission to
            # avoid the 'Permission denied' error:
            chmod_cmd = ['chmod', '+x', target_exe_path]

            # TODO(jroelofs): capture the copy_in and delete_remote commands,
            # and conjugate them with '&&'s around the first tuple element
            # returned here:
            return self._execute_command_remote(chmod_cmd + ['&&'] + cmd,
                                                target_cwd,
                                                env)
        finally:
            if target_cwd:
                self.delete_remote(target_cwd)

    def _execute_command_remote(self, cmd, remote_work_dir='.', env=None):
        raise NotImplementedError()


class SSHExecutor(RemoteExecutor):
    def __init__(self, host, username=None):
        super(SSHExecutor, self).__init__()

        self.user_prefix = username + '@' if username else ''
        self.host = host
        self.scp_command = 'scp'
        self.ssh_command = 'ssh'

        # TODO(jroelofs): switch this on some -super-verbose-debug config flag
        if False:
            self.local_run = tracing.trace_function(
                self.local_run, log_calls=True, log_results=True,
                label='ssh_local')

    def _remote_temp(self, is_dir):
        # TODO: detect what the target system is, and use the correct
        # mktemp command for it. (linux and darwin differ here, and I'm
        # sure windows has another way to do it)

        # Not sure how to do suffix on osx yet
        dir_arg = '-d' if is_dir else ''
        cmd = 'mktemp -q {} /tmp/libcxx.XXXXXXXXXX'.format(dir_arg)
        _, temp_path, err, exitCode = self._execute_command_remote([cmd])
        temp_path = temp_path.strip()
        if exitCode != 0:
            raise RuntimeError(err)
        return temp_path

    def _copy_in_file(self, src, dst):
        scp = self.scp_command
        remote = self.host
        remote = self.user_prefix + remote
        cmd = [scp, '-p', src, remote + ':' + dst]
        self.local_run(cmd)

    def _export_command(self, env):
        if not env:
            return []

        export_cmd = ['export']

        for k, v in env.items():
            v = v.replace('\\', '\\\\')
            if k == 'PATH':
                # Pick up the existing paths, so we don't lose any commands
                if self.target_info and self.target_info.is_windows():
                    export_cmd.append('PATH="%s;%PATH%"' % v)
                else:
                    export_cmd.append('PATH="%s:$PATH"' % v)
            else:
                export_cmd.append('"%s"="%s"' % (k, v))

        return export_cmd

    def _execute_command_remote(self, cmd, remote_work_dir='.', env=None):
        remote = self.user_prefix + self.host
        ssh_cmd = [self.ssh_command, '-oBatchMode=yes', remote]
        export_cmd = self._export_command(env)
        remote_cmd = ' '.join(cmd)
        if export_cmd:
            remote_cmd = ' '.join(export_cmd) + ' && ' + remote_cmd
        if remote_work_dir != '.':
            remote_cmd = 'cd ' + remote_work_dir + ' && ' + remote_cmd
        out, err, rc = self.local_run(ssh_cmd + [remote_cmd])
        return (remote_cmd, out, err, rc)
