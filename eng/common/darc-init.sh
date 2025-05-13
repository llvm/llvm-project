#!/usr/bin/env bash

source="${BASH_SOURCE[0]}"
darcVersion=''
versionEndpoint='https://maestro.dot.net/api/assets/darc-version?api-version=2020-02-20'
verbosity='minimal'

while [[ $# > 0 ]]; do
  opt="$(echo "$1" | tr "[:upper:]" "[:lower:]")"
  case "$opt" in
    --darcversion)
      darcVersion=$2
      shift
      ;;
    --versionendpoint)
      versionEndpoint=$2
      shift
      ;;
    --verbosity)
      verbosity=$2
      shift
      ;;
    --toolpath)
      toolpath=$2
      shift
      ;;
    *)
      echo "Invalid argument: $1"
      usage
      exit 1
      ;;
  esac

  shift
done

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

if [ -z "$darcVersion" ]; then
  darcVersion=$(curl -X GET "$versionEndpoint" -H "accept: text/plain")
fi

function InstallDarcCli {
  local darc_cli_package_name="microsoft.dotnet.darc"

  InitializeDotNetCli true
  local dotnet_root=$_InitializeDotNetCli

  if [ -z "$toolpath" ]; then
    local tool_list=$($dotnet_root/dotnet tool list -g)
    if [[ $tool_list = *$darc_cli_package_name* ]]; then
      echo $($dotnet_root/dotnet tool uninstall $darc_cli_package_name -g)
    fi
  else
    local tool_list=$($dotnet_root/dotnet tool list --tool-path "$toolpath")
    if [[ $tool_list = *$darc_cli_package_name* ]]; then
      echo $($dotnet_root/dotnet tool uninstall $darc_cli_package_name --tool-path "$toolpath")
    fi
  fi

  local arcadeServicesSource="https://pkgs.dev.azure.com/dnceng/public/_packaging/dotnet-eng/nuget/v3/index.json"

  echo "Installing Darc CLI version $darcVersion..."
  echo "You may need to restart your command shell if this is the first dotnet tool you have installed."
  if [ -z "$toolpath" ]; then
    echo $($dotnet_root/dotnet tool install $darc_cli_package_name --version $darcVersion --add-source "$arcadeServicesSource" -v $verbosity -g)
  else
    echo $($dotnet_root/dotnet tool install $darc_cli_package_name --version $darcVersion --add-source "$arcadeServicesSource" -v $verbosity --tool-path "$toolpath")
  fi
}

InstallDarcCli
