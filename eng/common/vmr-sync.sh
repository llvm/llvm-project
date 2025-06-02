#!/bin/bash

### This script is used for synchronizing the current repository into a local VMR.
### It pulls the current repository's code into the specified VMR directory for local testing or
### Source-Build validation.
###
### The tooling used for synchronization will clone the VMR repository into a temporary folder if
### it does not already exist. These clones can be reused in future synchronizations, so it is
### recommended to dedicate a folder for this to speed up re-runs.
###
### USAGE:
###   Synchronize current repository into a local VMR:
###     ./vmr-sync.sh --tmp "$HOME/repos/tmp" "$HOME/repos/dotnet"
###
### Options:
###   -t, --tmp, --tmp-dir PATH
###       Required. Path to the temporary folder where repositories will be cloned
###
###   -b, --branch, --vmr-branch BRANCH_NAME
###       Optional. Branch of the 'dotnet/dotnet' repo to synchronize. The VMR will be checked out to this branch
###
###   --debug
###       Optional. Turns on the most verbose logging for the VMR tooling
###
###   --remote name:URI
###       Optional. Additional remote to use during the synchronization
###       This can be used to synchronize to a commit from a fork of the repository
###       Example: 'runtime:https://github.com/yourfork/runtime'
###
###   --azdev-pat
###       Optional. Azure DevOps PAT to use for cloning private repositories.
###
###   -v, --vmr, --vmr-dir PATH
###       Optional. Path to the dotnet/dotnet repository. When null, gets cloned to the temporary folder

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

function print_help () {
    sed -n '/^### /,/^$/p' "$source" | cut -b 5-
}

COLOR_RED=$(tput setaf 1 2>/dev/null || true)
COLOR_CYAN=$(tput setaf 6 2>/dev/null || true)
COLOR_CLEAR=$(tput sgr0 2>/dev/null || true)
COLOR_RESET=uniquesearchablestring
FAILURE_PREFIX='> '

function fail () {
  echo "${COLOR_RED}$FAILURE_PREFIX${1//${COLOR_RESET}/${COLOR_RED}}${COLOR_CLEAR}" >&2
}

function highlight () {
  echo "${COLOR_CYAN}$FAILURE_PREFIX${1//${COLOR_RESET}/${COLOR_CYAN}}${COLOR_CLEAR}"
}

tmp_dir=''
vmr_dir=''
vmr_branch=''
additional_remotes=''
verbosity=verbose
azdev_pat=''
ci=false

while [[ $# -gt 0 ]]; do
  opt="$(echo "$1" | tr "[:upper:]" "[:lower:]")"
  case "$opt" in
    -t|--tmp|--tmp-dir)
      tmp_dir=$2
      shift
      ;;
    -v|--vmr|--vmr-dir)
      vmr_dir=$2
      shift
      ;;
    -b|--branch|--vmr-branch)
      vmr_branch=$2
      shift
      ;;
    --remote)
      additional_remotes="$additional_remotes $2"
      shift
      ;;
    --azdev-pat)
      azdev_pat=$2
      shift
      ;;
    --ci)
      ci=true
      ;;
    -d|--debug)
      verbosity=debug
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      fail "Invalid argument: $1"
      print_help
      exit 1
      ;;
  esac

  shift
done

# Validation

if [[ -z "$tmp_dir" ]]; then
  fail "Missing --tmp-dir argument. Please specify the path to the temporary folder where the repositories will be cloned"
  exit 1
fi

# Sanitize the input

if [[ -z "$vmr_dir" ]]; then
  vmr_dir="$tmp_dir/dotnet"
fi

if [[ ! -d "$tmp_dir" ]]; then
  mkdir -p "$tmp_dir"
fi

if [[ "$verbosity" == "debug" ]]; then
  set -x
fi

# Prepare the VMR

if [[ ! -d "$vmr_dir" ]]; then
  highlight "Cloning 'dotnet/dotnet' into $vmr_dir.."
  git clone https://github.com/dotnet/dotnet "$vmr_dir"

  if [[ -n "$vmr_branch" ]]; then
    git -C "$vmr_dir" switch -c "$vmr_branch"
  fi
else
  if ! git -C "$vmr_dir" diff --quiet; then
    fail "There are changes in the working tree of $vmr_dir. Please commit or stash your changes"
    exit 1
  fi

  if [[ -n "$vmr_branch" ]]; then
    highlight "Preparing $vmr_dir"
    git -C "$vmr_dir" checkout "$vmr_branch"
    git -C "$vmr_dir" pull
  fi
fi

set -e

# Prepare darc

highlight 'Installing .NET, preparing the tooling..'
source "./eng/common/tools.sh"
InitializeDotNetCli true
GetDarc
dotnetDir=$( cd ./.dotnet/; pwd -P )
dotnet=$dotnetDir/dotnet

highlight "Starting the synchronization of VMR.."
set +e

if [[ -n "$additional_remotes" ]]; then
  additional_remotes="--additional-remotes $additional_remotes"
fi

if [[ -n "$azdev_pat" ]]; then
  azdev_pat="--azdev-pat $azdev_pat"
fi

ci_arg=''
if [[ "$ci" == "true" ]]; then
  ci_arg="--ci"
fi

# Synchronize the VMR

export DOTNET_ROOT="$dotnetDir"

"$darc_tool" vmr forwardflow \
  --tmp "$tmp_dir"             \
  $azdev_pat                   \
  --$verbosity                 \
  $ci_arg                      \
  $additional_remotes          \
  "$vmr_dir"

if [[ $? == 0 ]]; then
  highlight "Synchronization succeeded"
else
  fail "Synchronization of repo to VMR failed!"
  fail "'$vmr_dir' is left in its last state (re-run of this script will reset it)."
  fail "Please inspect the logs which contain path to the failing patch file (use --debug to get all the details)."
  fail "Once you make changes to the conflicting VMR patch, commit it locally and re-run this script."
  exit 1
fi
