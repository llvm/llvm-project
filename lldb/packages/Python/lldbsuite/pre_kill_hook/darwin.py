"""Provides a pre-kill method to run on macOS."""
from __future__ import print_function

# system imports
import subprocess
import sys

# third-party module imports
import six


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

    output = subprocess.check_output(['sample', six.text_type(process_id),
                                      str(sample_time)])
    output_stream.write(output)
