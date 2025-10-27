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

# Enables an internal package source by name, if found. Returns 0 if found and enabled, 1 if not found.
EnableInternalPackageSource() {
    local PackageSourceName="$1"
    
    # Check if disabledPackageSources section exists
    grep -i "<disabledPackageSources>" "$ConfigFile" > /dev/null
    if [ "$?" != "0" ]; then
        return 1  # No disabled sources section
    fi
    
    # Check if this source name is disabled
    grep -i "<add key=\"$PackageSourceName\" value=\"true\"" "$ConfigFile" > /dev/null
    if [ "$?" == "0" ]; then
        echo "Enabling internal source '$PackageSourceName'."
        # Remove the disabled entry (including any surrounding comments or whitespace on the same line)
        sed -i.bak "/<add key=\"$PackageSourceName\" value=\"true\" \/>/d" "$ConfigFile"
        
        # Add the source name to PackageSources for credential handling
        PackageSources+=("$PackageSourceName")
        return 0  # Found and enabled
    fi
    
    return 1  # Not found in disabled sources
}

# Add source entry to PackageSources
AddPackageSource() {
    local SourceName="$1"
    local SourceEndPoint="$2"
    
    # Check if source already exists
    grep -i "<add key=\"$SourceName\"" "$ConfigFile" > /dev/null
    if [ "$?" == "0" ]; then
        echo "Package source $SourceName already present and enabled."
        PackageSources+=("$SourceName")
        return
    fi
    
    echo "Adding package source $SourceName"
    PackageSourcesNodeFooter="</packageSources>"
    PackageSourceTemplate="${TB}<add key=\"$SourceName\" value=\"$SourceEndPoint\" />"
    
    sed -i.bak "s|$PackageSourcesNodeFooter|$PackageSourceTemplate${NL}$PackageSourcesNodeFooter|" "$ConfigFile"
    PackageSources+=("$SourceName")
}

# Adds or enables the package source with the given name
AddOrEnablePackageSource() {
    local SourceName="$1"
    local SourceEndPoint="$2"
    
    # Try to enable if disabled, if not found then add new source
    EnableInternalPackageSource "$SourceName"
    if [ "$?" != "0" ]; then
        AddPackageSource "$SourceName" "$SourceEndPoint"
    fi
}

# Enable all darc-int package sources
EnableMaestroInternalPackageSources() {
    # Check if disabledPackageSources section exists
    grep -i "<disabledPackageSources>" "$ConfigFile" > /dev/null
    if [ "$?" != "0" ]; then
        return  # No disabled sources section
    fi
    
    # Find all darc-int disabled sources
    local DisabledDarcIntSources=()
    DisabledDarcIntSources+=$(grep -oh '"darc-int-[^"]*" value="true"' "$ConfigFile" | tr -d '"')
    
    for DisabledSourceName in ${DisabledDarcIntSources[@]} ; do
        if [[ $DisabledSourceName == darc-int* ]]; then
            EnableInternalPackageSource "$DisabledSourceName"
        fi
    done
}

# Ensure there is a <packageSources>...</packageSources> section.
grep -i "<packageSources>" $ConfigFile
if [ "$?" != "0" ]; then
    Write-PipelineTelemetryError -Category 'Build' "Error: Eng/common/SetupNugetSources.sh returned a non-zero exit code. NuGet config file must contain a packageSources section: $ConfigFile"
    ExitWithExitCode 1
fi

PackageSources=()

# Set feed suffix based on whether credentials are provided
FeedSuffix="v3/index.json"
if [ -n "$CredToken" ]; then
    FeedSuffix="v2"
    
    # Ensure there is a <packageSourceCredentials>...</packageSourceCredentials> section.
    grep -i "<packageSourceCredentials>" $ConfigFile
    if [ "$?" != "0" ]; then
        echo "Adding <packageSourceCredentials>...</packageSourceCredentials> section."

        PackageSourcesNodeFooter="</packageSources>"
        PackageSourceCredentialsTemplate="${TB}<packageSourceCredentials>${NL}${TB}</packageSourceCredentials>"

        sed -i.bak "s|$PackageSourcesNodeFooter|$PackageSourcesNodeFooter${NL}$PackageSourceCredentialsTemplate|" $ConfigFile
    fi
fi

# Check for disabledPackageSources; we'll enable any darc-int ones we find there
grep -i "<disabledPackageSources>" $ConfigFile > /dev/null
if [ "$?" == "0" ]; then
    echo "Checking for any darc-int disabled package sources in the disabledPackageSources node"
    EnableMaestroInternalPackageSources
fi

DotNetVersions=('5' '6' '7' '8' '9' '10')

for DotNetVersion in ${DotNetVersions[@]} ; do
    FeedPrefix="dotnet${DotNetVersion}";
    grep -i "<add key=\"$FeedPrefix\"" $ConfigFile > /dev/null
    if [ "$?" == "0" ]; then
        AddOrEnablePackageSource "$FeedPrefix-internal" "https://pkgs.dev.azure.com/dnceng/internal/_packaging/$FeedPrefix-internal/nuget/$FeedSuffix"
        AddOrEnablePackageSource "$FeedPrefix-internal-transport" "https://pkgs.dev.azure.com/dnceng/internal/_packaging/$FeedPrefix-internal-transport/nuget/$FeedSuffix"
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
            echo "	Inserting credential for feed: $FeedName"

            PackageSourceCredentialsNodeFooter="</packageSourceCredentials>"
            NewCredential="${TB}${TB}<$FeedName>${NL}${TB}<add key=\"Username\" value=\"dn-bot\" />${NL}${TB}${TB}<add key=\"ClearTextPassword\" value=\"$CredToken\" />${NL}${TB}${TB}</$FeedName>"

            sed -i.bak "s|$PackageSourceCredentialsNodeFooter|$NewCredential${NL}$PackageSourceCredentialsNodeFooter|" $ConfigFile
        fi
    done
fi
