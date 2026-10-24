# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shlex
import shutil
import sys

from . import config, options


def _resolve_ssaf_tool(
    cl_option, default_tool_name, cl_option_value, installed_shim_dir_path, errors
):
    error_prefix = "%s: " % cl_option
    words = (
        shlex.split(cl_option_value)
        if cl_option_value is not None
        else [default_tool_name]
    )
    if not words:
        errors.append(error_prefix + "nothing to run")
        return None
    if len(words) > 1:
        errors.append(
            error_prefix + "expected one executable, got '%s'" % cl_option_value
        )
        return None
    name_or_path = words[0]
    if os.sep in name_or_path:
        candidate = os.path.abspath(name_or_path)
    else:
        candidate = shutil.which(name_or_path)
    if not candidate:
        errors.append(error_prefix + "%s: not found on PATH" % name_or_path)
        return None
    if not os.path.isfile(candidate):
        errors.append(error_prefix + "not a file: %s" % candidate)
        return None
    if not os.access(candidate, os.X_OK):
        errors.append(error_prefix + "not executable: %s" % candidate)
        return None
    if os.path.dirname(os.path.realpath(candidate)) == installed_shim_dir_path:
        errors.append(
            error_prefix + "%s is one of our own shims; an earlier run's shim links "
            "are on PATH" % candidate
        )
        return None
    return candidate


def _may_be_cleared(output_dir):
    try:
        entries = os.listdir(output_dir)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return not entries or config.OUTPUT_DIR_MARKER in entries


def find_installed_shim_dir():
    entry = sys.modules.get("__main__")
    script = getattr(entry, "__file__", None) or sys.argv[0]
    script_dir = os.path.dirname(os.path.realpath(script))
    prefix = os.path.dirname(script_dir)
    installed = os.path.join(prefix, "libexec", "ssaf-orchestrator-shims")
    if os.path.isdir(installed):
        return installed
    return os.path.join(prefix, "shims")


def configure(command_line_opts, env):
    installed_shim_dir_path = find_installed_shim_dir()

    project_root = os.path.realpath(os.getcwd())
    output_dir = os.path.realpath(command_line_opts.output_dir)
    if project_root == output_dir or project_root.startswith(output_dir + os.sep):
        return None, options.Problem(
            options.ENVIRONMENT_ERROR,
            "--output-dir would contain the project, and its contents are "
            "deleted at the start of a run: %s" % output_dir,
        )
    if not _may_be_cleared(output_dir):
        return None, options.Problem(
            options.ENVIRONMENT_ERROR,
            "--output-dir is not empty and no earlier run made it, and its "
            "contents are deleted at the start of a run: %s" % output_dir,
        )

    errors = []

    ssaf_clang = _resolve_ssaf_tool(
        "--ssaf-clang",
        "clang",
        command_line_opts.ssaf_clang,
        installed_shim_dir_path,
        errors,
    )
    ssaf_linker = _resolve_ssaf_tool(
        "--ssaf-linker",
        "clang-ssaf-linker",
        command_line_opts.ssaf_linker,
        installed_shim_dir_path,
        errors,
    )
    ssaf_analyzer = (
        _resolve_ssaf_tool(
            "--ssaf-analyzer",
            "clang-ssaf-analyzer",
            command_line_opts.ssaf_analyzer,
            installed_shim_dir_path,
            errors,
        )
        if command_line_opts.analyses
        else None
    )
    ssaf_src_edit_merge = (
        _resolve_ssaf_tool(
            "--ssaf-src-edit-merge",
            "clang-ssaf-src-edit-merge",
            command_line_opts.ssaf_src_edit_merge,
            installed_shim_dir_path,
            errors,
        )
        if command_line_opts.transformation
        else None
    )
    ssaf_report_merge = (
        _resolve_ssaf_tool(
            "--ssaf-report-merge",
            "clang-ssaf-report-merge",
            command_line_opts.ssaf_report_merge,
            installed_shim_dir_path,
            errors,
        )
        if command_line_opts.transformation
        else None
    )
    ssaf_conflict_merge = (
        _resolve_ssaf_tool(
            "--ssaf-conflict-merge",
            "clang-ssaf-conflict-merge",
            command_line_opts.ssaf_conflict_merge,
            installed_shim_dir_path,
            errors,
        )
        if command_line_opts.transformation
        else None
    )

    if errors:
        return None, options.Problem(
            options.ENVIRONMENT_ERROR, "required tools not resolvable:", errors
        )

    common_pass_config = dict(
        ssaf_clang=ssaf_clang,
        ssaf_extra_clang_args=command_line_opts.ssaf_extra_clang_args,
    )
    workflow_config = config.SSAFWorkflowConfig(
        output_dir=output_dir,
        installed_shim_dir_path=installed_shim_dir_path,
        restored_path_env_var=env.get("PATH", ""),
        verbose=command_line_opts.verbose,
        build_command=command_line_opts.build_command,
        ssaf_analyzer=ssaf_analyzer,
        keep_going=command_line_opts.keep_going,
        build_environment=env,
    )
    analysis_config = config.SSAFAnalysisPassConfig(
        extractors=command_line_opts.extract_summaries,
        ssaf_linker=ssaf_linker,
        analyses=command_line_opts.analyses,
        **common_pass_config,
    )
    transformation_config = (
        config.SSAFTransformationPassConfig(
            ssaf_src_edit_merge=ssaf_src_edit_merge,
            ssaf_report_merge=ssaf_report_merge,
            ssaf_conflict_merge=ssaf_conflict_merge,
            transformation=command_line_opts.transformation,
            **common_pass_config,
        )
        if command_line_opts.transformation
        else None
    )
    return (workflow_config, analysis_config, transformation_config), None
