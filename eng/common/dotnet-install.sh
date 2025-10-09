#!/usr/bin/env bash

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

. "$scriptroot/tools.sh"

version='Latest'
architecture=''
runtime='dotnet'
runtimeSourceFeed=''
runtimeSourceFeedKey=''
while [[ $# -gt 0 ]]; do
  opt="$(echo "$1" | tr "[:upper:]" "[:lower:]")"
  case "$opt" in
    -version|-v)
      shift
      version="$1"
      ;;
    -architecture|-a)
      shift
      architecture="$1"
      ;;
    -runtime|-r)
      shift
      runtime="$1"
      ;;
    -runtimesourcefeed)
      shift
      runtimeSourceFeed="$1"
      ;;
    -runtimesourcefeedkey)
      shift
      runtimeSourceFeedKey="$1"
      ;;
    *)
      Write-PipelineTelemetryError -Category 'Build' -Message "Invalid argument: $1"
      exit 1
      ;;
  esac
  shift
done

# Use uname to determine what the CPU is, see https://en.wikipedia.org/wiki/Uname#Examples
cpuname=$(uname -m)
case $cpuname in
  arm64|aarch64)
    buildarch=arm64
    if [ "$(getconf LONG_BIT)" -lt 64 ]; then
        # This is 32-bit OS running on 64-bit CPU (for example Raspberry Pi OS)
        buildarch=arm
    fi
    ;;
  loongarch64)
    buildarch=loongarch64
    ;;
  amd64|x86_64)
    buildarch=x64
    ;;
  armv*l)
    buildarch=arm
    ;;
  i[3-6]86)
    buildarch=x86
    ;;
  riscv64)
    buildarch=riscv64
    ;;
  *)
    echo "Unknown CPU $cpuname detected, treating it as x64"
    buildarch=x64
    ;;
esac

dotnetRoot="${repo_root}.dotnet"
if [[ $architecture != "" ]] && [[ $architecture != $buildarch ]]; then
  dotnetRoot="$dotnetRoot/$architecture"
fi

InstallDotNet "$dotnetRoot" $version "$architecture" $runtime true $runtimeSourceFeed $runtimeSourceFeedKey || {
  local exit_code=$?
  Write-PipelineTelemetryError -Category 'InitializeToolset' -Message "dotnet-install.sh failed (exit code '$exit_code')." >&2
  ExitWithExitCode $exit_code
}

ExitWithExitCode 0
