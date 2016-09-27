"""Provides a pre-kill method to run on Linux.

This timeout pre-kill method relies on the Linux perf-tools
distribution.  The appropriate way to obtain this set of tools
will depend on the Linux distribution.

For Ubuntu 16.04, the invoke the following command:
sudo apt-get install perf-tools-unstable
"""
from __future__ import print_function

# system imports
import os
import subprocess
import sys
import tempfile


def do_pre_kill(process_id, runner_context, output_stream, sample_time=3):
    """Samples the given process id, and puts the output to output_stream.

    @param process_id the local process to sample.

    @param runner_context a dictionary of details about the architectures
    and platform on which the given process is running.  Expected keys are
    archs (array of architectures), platform_name, platform_url, and
    platform_working_dir.

    @param output_stream file-like object that should be used to write the
    results of sampling.

    @param sample_time specifies the time in seconds that should be captured.
    """

    # Validate args.
    if runner_context is None:
        raise Exception("runner_context argument is required")
    if not isinstance(runner_context, dict):
        raise Exception("runner_context argument must be a dictionary")

    # We will try to run sample on the local host only if there is no URL
    # to a remote.
    if "platform_url" in runner_context and (
            runner_context["platform_url"] is not None):
        import pprint
        sys.stderr.write(
            "warning: skipping timeout pre-kill sample invocation because we "
            "don't know how to run on a remote yet. runner_context={}\n"
            .format(pprint.pformat(runner_context)))

    # We're going to create a temp file, and immediately overwrite it with the
    # following command.  This just ensures we don't have any races in
    # creation of the temporary sample file.
    fileno, filename = tempfile.mkstemp(suffix='perfdata')
    os.close(fileno)
    fileno = None

    try:
        with open(os.devnull, 'w') as devnull:
            returncode = subprocess.call(['timeout', str(sample_time), 'perf',
                                          'record', '-g', '-o', filename, '-p', str(process_id)],
                                         stdout=devnull, stderr=devnull)
        if returncode == 0 or returncode == 124:
            # This is okay - this is the timeout return code, which is totally
            # expected.
            pass
        else:
            raise Exception("failed to call 'perf record .., error: {}".format(
                returncode))

        with open(os.devnull, 'w') as devnull:
            output = subprocess.check_output(['perf', 'report', '--call-graph',
                                              '--stdio', '-i', filename], stderr=devnull)
        output_stream.write(output)
    finally:
        os.remove(filename)
