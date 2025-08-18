#!/usr/bin/env bash

# This script adds internal feeds required to build commits that depend on internal package sources. For instance,
# dotnet6-internal would be added automatically if dotnet6 was found in the nuget.config file. In addition also enables
# disabled internal Maestro (darc-int*) feeds.
# 
# Optionally, this script also adds a credential entry for each of the internal feeds if supplied.
#
# See example call for this script below.
#
#  - task: Bash@3
#    displayName: Setup Internal Feeds
#    inputs:
#      filePath: $(System.DefaultWorkingDirectory)/eng/common/SetupNugetSources.sh
#      arguments: $(System.DefaultWorkingDirectory)/NuGet.config
#    condition: ne(variables['Agent.OS'], 'Windows_NT')
#  - task: NuGetAuthenticate@1
#
# Note that the NuGetAuthenticate task should be called after SetupNugetSources.
# This ensures that:
# - Appropriate creds are set for the added internal feeds (if not supplied to the scrupt)
# - The credential provider is installed.
#
# This logic is also abstracted into enable-internal-sources.yml.

ConfigFile=$1
CredToken=$2
NL='\n'
TB='    '

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

if [ ! -f "$ConfigFile" ]; then
    Write-PipelineTelemetryError -Category 'Build' "Error: Eng/common/SetupNugetSources.sh returned a non-zero exit code. Couldn't find the NuGet config file: $ConfigFile"
    ExitWithExitCode 1
fi

if [[ `uname -s` == "Darwin" ]]; then
    NL=$'\\\n'
    TB=''
fi

# Ensure there is a <packageSources>...</packageSources> section.
grep -i "<packageSources>" $ConfigFile
if [ "$?" != "0" ]; then
    echo "Adding <packageSources>...</packageSources> section."
    ConfigNodeHeader="<configuration>"
    PackageSourcesTemplate="${TB}<packageSources>${NL}${TB}</packageSources>"

    sed -i.bak "s|$ConfigNodeHeader|$ConfigNodeHeader${NL}$PackageSourcesTemplate|" $ConfigFile
fi

# Ensure there is a <packageSourceCredentials>...</packageSourceCredentials> section. 
grep -i "<packageSourceCredentials>" $ConfigFile
if [ "$?" != "0" ]; then
    echo "Adding <packageSourceCredentials>...</packageSourceCredentials> section."

    PackageSourcesNodeFooter="</packageSources>"
    PackageSourceCredentialsTemplate="${TB}<packageSourceCredentials>${NL}${TB}</packageSourceCredentials>"

    sed -i.bak "s|$PackageSourcesNodeFooter|$PackageSourcesNodeFooter${NL}$PackageSourceCredentialsTemplate|" $ConfigFile
fi

PackageSources=()

# Ensure dotnet3.1-internal and dotnet3.1-internal-transport are in the packageSources if the public dotnet3.1 feeds are present
grep -i "<add key=\"dotnet3.1\"" $ConfigFile
if [ "$?" == "0" ]; then
    grep -i "<add key=\"dotnet3.1-internal\"" $ConfigFile
    if [ "$?" != "0" ]; then
        echo "Adding dotnet3.1-internal to the packageSources."
        PackageSourcesNodeFooter="</packageSources>"
        PackageSourceTemplate="${TB}<add key=\"dotnet3.1-internal\" value=\"https://pkgs.dev.azure.com/dnceng/_packaging/dotnet3.1-internal/nuget/v2\" />"

        sed -i.bak "s|$PackageSourcesNodeFooter|$PackageSourceTemplate${NL}$PackageSourcesNodeFooter|" $ConfigFile
    fi
    PackageSources+=('dotnet3.1-internal')

    grep -i "<add key=\"dotnet3.1-internal-transport\">" $ConfigFile
    if [ "$?" != "0" ]; then
        echo "Adding dotnet3.1-internal-transport to the packageSources."
        PackageSourcesNodeFooter="</packageSources>"
        PackageSourceTemplate="${TB}<add key=\"dotnet3.1-internal-transport\" value=\"https://pkgs.dev.azure.com/dnceng/_packaging/dotnet3.1-internal-transport/nuget/v2\" />"

        sed -i.bak "s|$PackageSourcesNodeFooter|$PackageSourceTemplate${NL}$PackageSourcesNodeFooter|" $ConfigFile
    fi
    PackageSources+=('dotnet3.1-internal-transport')
fi

DotNetVersions=('5' '6' '7' '8' '9')

for DotNetVersion in ${DotNetVersions[@]} ; do
    FeedPrefix="dotnet${DotNetVersion}";
    grep -i "<add key=\"$FeedPrefix\"" $ConfigFile
    if [ "$?" == "0" ]; then
        grep -i "<add key=\"$FeedPrefix-internal\"" $ConfigFile
        if [ "$?" != "0" ]; then
            echo "Adding $FeedPrefix-internal to the packageSources."
            PackageSourcesNodeFooter="</packageSources>"
            PackageSourceTemplate="${TB}<add key=\"$FeedPrefix-internal\" value=\"https://pkgs.dev.azure.com/dnceng/internal/_packaging/$FeedPrefix-internal/nuget/v2\" />"

            sed -i.bak "s|$PackageSourcesNodeFooter|$PackageSourceTemplate${NL}$PackageSourcesNodeFooter|" $ConfigFile
        fi
        PackageSources+=("$FeedPrefix-internal")

        grep -i "<add key=\"$FeedPrefix-internal-transport\">" $ConfigFile
        if [ "$?" != "0" ]; then
            echo "Adding $FeedPrefix-internal-transport to the packageSources."
            PackageSourcesNodeFooter="</packageSources>"
            PackageSourceTemplate="${TB}<add key=\"$FeedPrefix-internal-transport\" value=\"https://pkgs.dev.azure.com/dnceng/internal/_packaging/$FeedPrefix-internal-transport/nuget/v2\" />"

            sed -i.bak "s|$PackageSourcesNodeFooter|$PackageSourceTemplate${NL}$PackageSourcesNodeFooter|" $ConfigFile
        fi
        PackageSources+=("$FeedPrefix-internal-transport")
    fi
done

# I want things split line by line
PrevIFS=$IFS
IFS=$'\n'
PackageSources+="$IFS"
PackageSources+=$(grep -oh '"darc-int-[^"]*"' $ConfigFile | tr -d '"')
IFS=$PrevIFS

if [ "$CredToken" ]; then
    for FeedName in ${PackageSources[@]} ; do
        # Check if there is no existing credential for this FeedName
        grep -i "<$FeedName>" $ConfigFile 
        if [ "$?" != "0" ]; then
            echo "Adding credentials for $FeedName."

            PackageSourceCredentialsNodeFooter="</packageSourceCredentials>"
            NewCredential="${TB}${TB}<$FeedName>${NL}<add key=\"Username\" value=\"dn-bot\" />${NL}<add key=\"ClearTextPassword\" value=\"$CredToken\" />${NL}</$FeedName>"

            sed -i.bak "s|$PackageSourceCredentialsNodeFooter|$NewCredential${NL}$PackageSourceCredentialsNodeFooter|" $ConfigFile
        fi
    done
fi

# Re-enable any entries in disabledPackageSources where the feed name contains darc-int
grep -i "<disabledPackageSources>" $ConfigFile
if [ "$?" == "0" ]; then
    DisabledDarcIntSources=()
    echo "Re-enabling any disabled \"darc-int\" package sources in $ConfigFile"
    DisabledDarcIntSources+=$(grep -oh '"darc-int-[^"]*" value="true"' $ConfigFile  | tr -d '"')
    for DisabledSourceName in ${DisabledDarcIntSources[@]} ; do
        if [[ $DisabledSourceName == darc-int* ]]
            then
                OldDisableValue="<add key=\"$DisabledSourceName\" value=\"true\" />"
                NewDisableValue="<!-- Reenabled for build : $DisabledSourceName -->"
                sed -i.bak "s|$OldDisableValue|$NewDisableValue|" $ConfigFile
                echo "Neutralized disablePackageSources entry for '$DisabledSourceName'"
        fi
    done
fi
