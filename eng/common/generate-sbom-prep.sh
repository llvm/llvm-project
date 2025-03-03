#!/usr/bin/env bash

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
. $scriptroot/pipeline-logging-functions.sh


# replace all special characters with _, some builds use special characters like : in Agent.Jobname, that is not a permissible name while uploading artifacts.
artifact_name=$SYSTEM_STAGENAME"_"$AGENT_JOBNAME"_SBOM"
safe_artifact_name="${artifact_name//["/:<>\\|?@*$" ]/_}"
manifest_dir=$1

# Normally - we'd listen to the manifest path given, but 1ES templates will overwrite if this level gets uploaded directly
# with their own overwriting ours. So we create it as a sub directory of the requested manifest path.
sbom_generation_dir="$manifest_dir/$safe_artifact_name"

if [ ! -d "$sbom_generation_dir" ] ; then
  mkdir -p "$sbom_generation_dir"
  echo "Sbom directory created." $sbom_generation_dir
else
  Write-PipelineTelemetryError -category 'Build'  "Unable to create sbom folder."
fi

echo "Artifact name before : "$artifact_name
echo "Artifact name after : "$safe_artifact_name
export ARTIFACT_NAME=$safe_artifact_name
echo "##vso[task.setvariable variable=ARTIFACT_NAME]$safe_artifact_name"

exit 0
