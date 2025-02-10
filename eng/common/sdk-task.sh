#!/usr/bin/env bash

show_usage() {
    echo "Common settings:"
    echo "  --task <value>           Name of Arcade task (name of a project in SdkTasks directory of the Arcade SDK package)"
    echo "  --restore                Restore dependencies"
    echo "  --verbosity <value>      Msbuild verbosity: q[uiet], m[inimal], n[ormal], d[etailed], and diag[nostic]"
    echo "  --help                   Print help and exit"
    echo ""
    echo "Command line arguments not listed above are passed thru to msbuild."
}

source="${BASH_SOURCE[0]}"

# resolve $source until the file is no longer a symlink
while [[ -h "$source" ]]; do
  scriptroot="$( cd -P "$( dirname "$source" )" && pwd )"
  source="$(readlink "$source")"
  # if $source was a relative symlink, we need to resolve it relative to the path where the
  # symlink file was located
  [[ $source != /* ]] && source="$scriptroot/$source"
done
scriptroot="$( cd -P "$( dirname "$source" )" && pwd )"

Build() {
    local target=$1
    local log_suffix=""
    [[ "$target" != "Execute" ]] && log_suffix=".$target"
    local log="$log_dir/$task$log_suffix.binlog"
    local output_path="$toolset_dir/$task/"

    MSBuild "$taskProject" \
        /bl:"$log" \
        /t:"$target" \
        /p:Configuration="$configuration" \
        /p:RepoRoot="$repo_root" \
        /p:BaseIntermediateOutputPath="$output_path" \
        /v:"$verbosity" \
        $properties
}

configuration="Debug"
verbosity="minimal"
restore=false
help=false
properties=''

while (($# > 0)); do
  lowerI="$(echo $1 | tr "[:upper:]" "[:lower:]")"
  case $lowerI in
    --task)
      task=$2
      shift 2
      ;;
    --restore)
      restore=true
      shift 1
      ;;
    --verbosity)
      verbosity=$2
      shift 2
      ;;
    --help)
      help=true
      shift 1
      ;;
    *)
      properties="$properties $1"
      shift 1
      ;;
  esac
done

ci=true
binaryLog=true
warnAsError=true

if $help; then
  show_usage
  exit 0
fi

. "$scriptroot/tools.sh"
InitializeToolset

if [[ -z "$task" ]]; then
    Write-PipelineTelemetryError -Category 'Task' -Name 'MissingTask' -Message "Missing required parameter '-task <value>'"
    ExitWithExitCode 1
fi

taskProject=$(GetSdkTaskProject "$task")
if [[ ! -e "$taskProject" ]]; then
    Write-PipelineTelemetryError -Category 'Task' -Name 'UnknownTask' -Message "Unknown task: $task"
    ExitWithExitCode 1
fi

if $restore; then
    Build "Restore"
fi

Build "Execute"


ExitWithExitCode 0
