Param(
    [Parameter(Mandatory=$true)][string] $ManifestDirPath    # Manifest directory where sbom will be placed
)

. $PSScriptRoot\pipeline-logging-functions.ps1

# Normally - we'd listen to the manifest path given, but 1ES templates will overwrite if this level gets uploaded directly
# with their own overwriting ours. So we create it as a sub directory of the requested manifest path.
$ArtifactName = "${env:SYSTEM_STAGENAME}_${env:AGENT_JOBNAME}_SBOM"
$SafeArtifactName = $ArtifactName -replace '["/:<>\\|?@*"() ]', '_'
$SbomGenerationDir = Join-Path $ManifestDirPath $SafeArtifactName

Write-Host "Artifact name before : $ArtifactName"
Write-Host "Artifact name after : $SafeArtifactName"

Write-Host "Creating dir $ManifestDirPath"

# create directory for sbom manifest to be placed
if (!(Test-Path -path $SbomGenerationDir))
{
  New-Item -ItemType Directory -path $SbomGenerationDir
  Write-Host "Successfully created directory $SbomGenerationDir"
}
else{
  Write-PipelineTelemetryError -category 'Build'  "Unable to create sbom folder."
}

Write-Host "Updating artifact name"
Write-Host "##vso[task.setvariable variable=ARTIFACT_NAME]$SafeArtifactName"
