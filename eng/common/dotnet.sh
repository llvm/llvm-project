#!/usr/bin/env bash

# This script is used to install the .NET SDK.
# It will also invoke the SDK with any provided arguments.

source="${BASH_SOURCE[0]}"
# resolve $SOURCE until the file is no longer a symlink
while [[ -h $source ]]; do
  scriptroot="$( cd -P "$( dirname "$source" )" && pwd )"
  source="$(readlink "$source")"

  # if $source was a relative symlink, we need to resolve it relative to the path where the
  # symlink file was located
  [[ $source != /* ]] && source="$scriptroot/$source"
done
scriptroot="$( cd -P "$( dirname "$source" )" && pwd )"

source $scriptroot/tools.sh
InitializeDotNetCli true # install

# Invoke acquired SDK with args if they are provided
if [[ $# -gt 0 ]]; then
  __dotnetDir=${_InitializeDotNetCli}
  dotnetPath=${__dotnetDir}/dotnet
  ${dotnetPath} "$@"
fi
